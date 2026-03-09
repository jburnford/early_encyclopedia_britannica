#!/usr/bin/env python3
"""
Run EarlyModernNER on Encyclopedia Britannica articles.

Extracts TOPONYM, PERSON, ORGANIZATION, and COMMODITY entities using
four specialized LoRA adapters on Qwen3-4B (4-bit NF4 quantized).

Processes one edition at a time with checkpoint-based resume capability.
Each entity type adapter is loaded sequentially to minimize GPU memory.

Usage:
    python graphrag/run_ner.py --edition-year 1771
    python graphrag/run_ner.py --edition-year 1771 --batch-size 200
    python graphrag/run_ner.py --edition-year 1771 --entity-types TOPONYM PERSON
    python graphrag/run_ner.py --edition-year 1771 --max-articles 100  # quick test
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from earlymodernner.pipeline import (
    load_model_with_adapter,
    run_entity_extraction,
    merge_entity_results,
    get_adapter_path,
)
from earlymodernner.constants import ENTITY_TYPES

# Defaults (match earlymodernner internals)
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_NAMES = {
    "TOPONYM": "toponym_lora",
    "PERSON": "person_lora",
    "ORGANIZATION": "organization_lora",
    "COMMODITY": "commodity_lora",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "data" / "export"
NER_DIR = REPO_ROOT / "data" / "ner"

# Skip trivial cross-references ("See X.")
MIN_WORDS = 10

# Chunk long articles to prevent KV cache OOM on small GPUs.
# 3000 words ≈ 4K tokens → safe for 20GB MIG slices.
DEFAULT_MAX_WORDS = 3000


def find_export_file(edition_year: int) -> Path:
    """Find the export JSONL file for a given edition year."""
    matches = list(EXPORT_DIR.glob(f"eb_*_{edition_year}.jsonl"))
    if len(matches) == 1:
        return matches[0]
    elif not matches:
        sys.exit(f"Error: No export file for {edition_year} in {EXPORT_DIR}")
    else:
        sys.exit(f"Error: Multiple files for {edition_year}: {matches}")


def load_articles(edition_year: int, max_articles: int = None,
                   volume: int = None) -> list[dict]:
    """Load articles from export JSONL, filtering trivial entries."""
    export_file = find_export_file(edition_year)
    vol_label = f" vol {volume}" if volume is not None else ""
    print(f"Reading {export_file.name}{vol_label}...")
    articles = []
    skipped = 0
    vol_skipped = 0
    with open(export_file) as f:
        for line in f:
            art = json.loads(line)
            if volume is not None and art.get("volume") != volume:
                vol_skipped += 1
                continue
            if art.get("word_count", 0) < MIN_WORDS:
                skipped += 1
                continue
            articles.append(art)
            if max_articles and len(articles) >= max_articles:
                break
    print(f"  {len(articles):,} articles loaded (skipped {skipped:,} with <{MIN_WORDS} words)")
    if volume is not None:
        print(f"  (filtered to volume {volume}, skipped {vol_skipped:,} from other volumes)")
    return articles


def list_volumes(edition_year: int) -> list[int]:
    """List all volume numbers in an edition's export file."""
    export_file = find_export_file(edition_year)
    vols = set()
    with open(export_file) as f:
        for line in f:
            art = json.loads(line)
            v = art.get("volume")
            if v is not None:
                vols.add(v)
    return sorted(vols)


def load_checkpoint(path: Path) -> dict:
    """Load checkpoint: {entity_type: {doc_id: [entities]}}."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, data: dict):
    """Atomic checkpoint save via temp file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(path)


CHUNK_OVERLAP = 200  # words of overlap between chunks


def chunk_text(text: str, max_words: int) -> list[str]:
    """Split text into overlapping chunks of max_words.

    Long articles are split into windows with CHUNK_OVERLAP words of
    overlap so entities near chunk boundaries aren't missed.
    Returns a list of text chunks (one element for short articles).
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    stride = max_words - CHUNK_OVERLAP
    for start in range(0, len(words), stride):
        chunk_words = words[start:start + max_words]
        if len(chunk_words) < MIN_WORDS:
            break
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def run_ner_edition(edition_year: int, batch_size: int = 500,
                    entity_types: list[str] = None, max_articles: int = None,
                    max_words: int = DEFAULT_MAX_WORDS, verbose: bool = False,
                    volume: int = None):
    """Run NER extraction on all articles for one edition (or one volume)."""
    if entity_types is None:
        entity_types = list(ENTITY_TYPES)

    NER_DIR.mkdir(parents=True, exist_ok=True)

    # Load articles and chunk long ones
    articles = load_articles(edition_year, max_articles, volume=volume)
    chunked_count = 0
    total_chunks = 0
    documents = []
    chunk_map = {}  # chunk_doc_id -> original article_id
    for art in articles:
        text = art.get("text", "").strip()
        if not text:
            continue
        aid = art["article_id"]
        chunks = chunk_text(text, max_words) if max_words else [text]
        if len(chunks) > 1:
            chunked_count += 1
            total_chunks += len(chunks)
            for ci, chunk in enumerate(chunks):
                chunk_id = f"{aid}__chunk_{ci}"
                documents.append({"doc_id": chunk_id, "text": chunk})
                chunk_map[chunk_id] = aid
        else:
            documents.append({"doc_id": aid, "text": chunks[0]})
    if chunked_count:
        print(f"  Chunked {chunked_count} long articles into {total_chunks} chunks "
              f"({max_words} words, {CHUNK_OVERLAP} overlap)")
    all_doc_ids = {d["doc_id"] for d in documents}
    print(f"  {len(documents):,} documents ready for NER\n")

    # Checkpoint for resume (volume-specific if filtering by volume)
    vol_suffix = f"_v{volume}" if volume is not None else ""
    ckpt_path = NER_DIR / f".checkpoint_{edition_year}{vol_suffix}.json"
    checkpoint = load_checkpoint(ckpt_path)

    # Process each entity type with its adapter
    all_results = {}  # {entity_type: {doc_id: [entities]}}
    overall_start = time.time()

    for entity_type in entity_types:
        print(f"{'='*60}")
        print(f"  {entity_type}")
        print(f"{'='*60}")

        # Check checkpoint for this entity type
        # Checkpoint may contain chunk IDs or original article IDs from previous
        # (pre-chunking) runs. If an original article_id exists in the checkpoint,
        # treat all its chunks as done too.
        if entity_type in checkpoint:
            done_ids = set(checkpoint[entity_type].keys())
            # Map old unchunked article_ids to cover their chunks
            done_orig_ids = {chunk_map.get(k, k) for k in done_ids} | done_ids
            remaining = [d for d in documents
                         if d["doc_id"] not in done_ids
                         and chunk_map.get(d["doc_id"], d["doc_id"]) not in done_ids]
            all_results[entity_type] = {
                k: v for k, v in checkpoint[entity_type].items()
                if k in all_doc_ids or k in {chunk_map.get(did, did) for did in all_doc_ids}
            }
            if not remaining:
                print(f"  Complete ({len(done_ids):,} articles cached)\n")
                continue
            print(f"  Resuming: {len(done_ids):,} done, {len(remaining):,} remaining")
        else:
            remaining = documents
            checkpoint[entity_type] = {}
            all_results[entity_type] = {}

        # Load model + adapter
        adapter_name = ADAPTER_NAMES[entity_type]
        print(f"  Loading adapter: {adapter_name}...")
        t0 = time.time()
        adapter_path = get_adapter_path(adapter_name)
        model, tokenizer = load_model_with_adapter(BASE_MODEL, str(adapter_path))
        print(f"  Loaded in {time.time() - t0:.1f}s")

        # Process in batches with checkpointing
        n_entities = 0
        t_start = time.time()
        n_batches = (len(remaining) + batch_size - 1) // batch_size

        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]
            batch_num = i // batch_size + 1

            print(f"  Batch {batch_num}/{n_batches} "
                  f"({len(batch)} articles)...", end=" ", flush=True)
            t_batch = time.time()

            results = run_entity_extraction(
                model, tokenizer, batch, entity_type, verbose=verbose
            )

            # Accumulate
            batch_ents = 0
            for doc_id, entities in results.items():
                all_results[entity_type][doc_id] = entities
                checkpoint[entity_type][doc_id] = entities
                batch_ents += len(entities)
            n_entities += batch_ents

            elapsed = time.time() - t_batch
            rate = len(batch) / elapsed if elapsed > 0 else 0
            print(f"{elapsed:.0f}s ({rate:.2f} art/s), {batch_ents} entities")

            save_checkpoint(ckpt_path, checkpoint)

        # Free GPU memory before loading next adapter
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        elapsed_total = time.time() - t_start
        print(f"  {entity_type} done: {n_entities:,} entities in {elapsed_total:.0f}s\n")

    # Reassemble chunks back into original article IDs
    if chunk_map:
        print(f"{'='*60}")
        print("  Reassembling chunks into articles")
        print(f"{'='*60}")
        for entity_type in all_results:
            reassembled = {}
            for doc_id, entities in all_results[entity_type].items():
                orig_id = chunk_map.get(doc_id, doc_id)
                if orig_id not in reassembled:
                    reassembled[orig_id] = []
                reassembled[orig_id].extend(entities)
            # Deduplicate: same entity text + type from overlapping chunks
            for orig_id in reassembled:
                seen = set()
                deduped = []
                for ent in reassembled[orig_id]:
                    key = (ent["text"], ent.get("type", entity_type))
                    if key not in seen:
                        seen.add(key)
                        deduped.append(ent)
                reassembled[orig_id] = deduped
            all_results[entity_type] = reassembled
        # Rebuild documents list with original IDs for merge step
        seen_ids = set()
        orig_documents = []
        for d in documents:
            orig_id = chunk_map.get(d["doc_id"], d["doc_id"])
            if orig_id not in seen_ids:
                seen_ids.add(orig_id)
                # Use full original text for the merge
                art_text = next(
                    (a["text"] for a in articles if a["article_id"] == orig_id), ""
                )
                orig_documents.append({"doc_id": orig_id, "text": art_text})
        documents = orig_documents
        print(f"  {len(documents):,} articles after reassembly\n")

    # Merge using priority cascade (TOPONYM > COMMODITY > PERSON > ORGANIZATION)
    print(f"{'='*60}")
    print("  Merging (priority: TOPONYM > COMMODITY > PERSON > ORGANIZATION)")
    print(f"{'='*60}")
    merged = merge_entity_results(all_results, documents)

    # Build article metadata lookup
    meta = {art["article_id"]: art for art in articles}

    # Determine output filename from article metadata
    edition_label = articles[0]["edition"] if articles else str(edition_year)
    output_file = NER_DIR / f"eb_{edition_label}_{edition_year}{vol_suffix}.entities.jsonl"

    with open(output_file, "w") as f:
        for doc in merged:
            art = meta.get(doc["doc_id"], {})
            record = {
                "article_id": doc["doc_id"],
                "title": art.get("title", ""),
                "edition": art.get("edition", edition_label),
                "edition_year": edition_year,
                "volume": art.get("volume"),
                "entities": doc["entities"],
                "entity_counts": doc["entity_counts"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary
    total_time = time.time() - overall_start
    total_ents = sum(
        sum(doc["entity_counts"].get(et, 0) for et in ENTITY_TYPES)
        for doc in merged
    )
    articles_with_ents = sum(1 for doc in merged if doc["entities"])

    print(f"\n{'='*60}")
    print(f"  RESULTS: {output_file.name}")
    print(f"{'='*60}")
    print(f"  Articles processed:     {len(merged):,}")
    print(f"  Articles with entities: {articles_with_ents:,} "
          f"({100 * articles_with_ents / len(merged):.1f}%)")
    print(f"  Total entities:         {total_ents:,}")
    for et in ENTITY_TYPES:
        count = sum(doc["entity_counts"].get(et, 0) for doc in merged)
        print(f"    {et:15s}: {count:,}")
    print(f"  Total time:  {total_time / 60:.1f} minutes")
    print(f"  Rate:        {len(merged) / total_time:.2f} articles/sec")

    # Clean up checkpoint on success
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"\nDone. Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run EarlyModernNER on Encyclopedia Britannica articles"
    )
    parser.add_argument(
        "--edition-year", type=int, required=True,
        choices=[1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860],
        help="Edition year to process",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Articles per checkpoint save (default: 500)",
    )
    parser.add_argument(
        "--entity-types", nargs="+", default=None,
        choices=ENTITY_TYPES,
        help="Entity types to extract (default: all four)",
    )
    parser.add_argument(
        "--max-articles", type=int, default=None,
        help="Limit number of articles (for testing)",
    )
    parser.add_argument(
        "--max-words", type=int, default=DEFAULT_MAX_WORDS,
        help=f"Chunk articles longer than N words (default: {DEFAULT_MAX_WORDS})",
    )
    parser.add_argument(
        "--volume", type=int, default=None,
        help="Process only this volume number",
    )
    parser.add_argument(
        "--list-volumes", action="store_true",
        help="List available volumes for the edition and exit",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-article extraction details",
    )
    args = parser.parse_args()

    if args.list_volumes:
        vols = list_volumes(args.edition_year)
        print(f"Volumes for {args.edition_year}: {vols}")
        return

    run_ner_edition(
        args.edition_year,
        batch_size=args.batch_size,
        entity_types=args.entity_types,
        max_articles=args.max_articles,
        max_words=args.max_words,
        verbose=args.verbose,
        volume=args.volume,
    )


if __name__ == "__main__":
    main()
