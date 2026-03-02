"""Phase 2: Async batched LLM classification of paragraphs."""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import aiohttp

from config import (
    API_URL, MODEL, BATCH_SIZE, OVERLAP, STEP_SIZE,
    MAX_CONCURRENT, REQUEST_TIMEOUT, MAX_RETRIES,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    INPUT_DIR, PARAGRAPHS_DIR, CLASSIFICATIONS_DIR,
    ensure_dirs,
)

log = logging.getLogger(__name__)

# JSON schema for guided decoding — forces structured output, no thinking tokens
GUIDED_JSON_SCHEMA = json.dumps({
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "i": {"type": "integer"},
            "type": {
                "type": "string",
                "enum": [
                    "article_start", "subsection_start", "running_header",
                    "cross_reference", "front_matter", "back_matter",
                    "author_attribution", "footnote_sep",
                ],
            },
            "title": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "target": {"type": "string"},
        },
        "required": ["i", "type"],
    },
})


def build_prompt(paragraphs: list[dict], edition_name: str, year: int, volume: int) -> str:
    """Build the classification prompt for a batch of paragraphs."""
    para_text = ""
    for p in paragraphs:
        idx = p["index"]
        preview = p["preview"].replace("\n", " ")
        para_text += f"[P{idx}] {preview}\n\n"

    return f"""Classify these Encyclopedia Britannica ({edition_name} edition, {year}) paragraphs from volume {volume}. Report ONLY non-body paragraphs.

Types:
- "article_start": New encyclopedia article (title usually ALL-CAPS). Include "title" and "keywords" (2-5 topic words).
- "subsection_start": Section within a long article (e.g., "Part I.", "CHAPTER II."). Include "title".
- "running_header": Short ALL-CAPS OCR page header artifact interrupting text mid-sentence.
- "cross_reference": Short "X. See Y." redirect entry. Include "title" and "target" (the article name being referenced).
- "front_matter": Title page, preface, or dedication at start of volume.
- "back_matter": Material after "END OF VOLUME" — plate directions, errata.
- "author_attribution": Author initials in parentheses at end of article, e.g. "(C. M.—L.)"
- "footnote_sep": A "---" separator line.

If ALL paragraphs are body text, output: []

=== PARAGRAPHS ===
{para_text.rstrip()}"""


def parse_llm_response(content: str) -> list[dict]:
    """Extract JSON array from LLM response, handling thinking tokens and truncation."""
    # DeepSeek-R1 may include <think>...</think> before the answer
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Try direct parse first
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from surrounding text
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Handle truncated JSON arrays (hit max_tokens mid-output)
    # Try to salvage complete objects from a truncated array
    if content.lstrip().startswith("["):
        # Find all complete JSON objects in the truncated array
        salvaged = []
        for obj_match in re.finditer(r'\{[^{}]*\}', content):
            try:
                obj = json.loads(obj_match.group())
                if "i" in obj and "type" in obj:
                    salvaged.append(obj)
            except json.JSONDecodeError:
                continue
        if salvaged:
            log.debug(f"Salvaged {len(salvaged)} objects from truncated JSON")
            return salvaged

    # Empty-looking content
    if content.strip() in ("[]", "", "null", "None"):
        return []

    log.warning(f"Could not parse LLM response: {content[:200]}")
    return []


async def classify_batch(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_url: str,
    paragraphs: list[dict],
    edition_name: str,
    year: int,
    volume: int,
    batch_idx: int,
) -> tuple[int, list[dict]]:
    """Send one batch to the LLM and return classifications."""
    prompt = build_prompt(paragraphs, edition_name, year, volume)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "guided_json": GUIDED_JSON_SCHEMA,
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.post(
                    api_url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning(f"Batch {batch_idx} attempt {attempt+1}: "
                                    f"HTTP {resp.status}: {body[:200]}")
                        await asyncio.sleep(2 ** attempt)
                        continue

                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    classifications = parse_llm_response(content)
                    return batch_idx, classifications

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning(f"Batch {batch_idx} attempt {attempt+1}: "
                        f"{type(e).__name__}: {e}")
            await asyncio.sleep(2 ** attempt)

    log.error(f"Batch {batch_idx}: all {MAX_RETRIES} attempts failed")
    return batch_idx, []


def make_windows(paragraphs: list[dict]) -> list[list[dict]]:
    """Create sliding windows of paragraphs with overlap."""
    windows = []
    i = 0
    while i < len(paragraphs):
        window = paragraphs[i : i + BATCH_SIZE]
        windows.append(window)
        i += STEP_SIZE
        # Don't create a tiny trailing window
        if i < len(paragraphs) and len(paragraphs) - i < OVERLAP:
            break
    return windows


def merge_classifications(
    windows: list[list[dict]],
    results: dict[int, list[dict]],
) -> dict[int, dict]:
    """Merge overlapping window results, deduplicating by paragraph index.

    For paragraphs in overlap zones, prefer the classification from the window
    where the paragraph is more central (not at the edge).
    """
    classifications = {}  # para_index -> classification dict

    for win_idx, window in enumerate(windows):
        batch_results = results.get(win_idx, [])
        window_indices = {p["index"] for p in window}
        window_start = window[0]["index"]
        window_end = window[-1]["index"]
        window_mid = (window_start + window_end) / 2

        for cls in batch_results:
            para_idx = cls.get("i")
            if para_idx is None or para_idx not in window_indices:
                continue

            if para_idx not in classifications:
                classifications[para_idx] = cls
            else:
                # Prefer the window where this paragraph is more central
                existing_win = classifications[para_idx].get("_win", 0)
                existing_window = windows[existing_win]
                existing_mid = (existing_window[0]["index"] + existing_window[-1]["index"]) / 2
                if abs(para_idx - window_mid) < abs(para_idx - existing_mid):
                    classifications[para_idx] = cls

            classifications[para_idx]["_win"] = win_idx

    # Remove internal _win field
    for cls in classifications.values():
        cls.pop("_win", None)

    return classifications


async def classify_file(
    input_path: Path,
    para_path: Path,
    api_url: str,
) -> Path:
    """Classify all paragraphs in a single file."""
    stem = input_path.stem
    output_path = CLASSIFICATIONS_DIR / f"{stem}.classifications.jsonl"

    # Load metadata from source JSONL
    with open(input_path) as f:
        meta = json.loads(f.readline())
    edition_name = meta["edition_name"]
    year = meta["edition"]
    volume = meta["volume"]

    # Load paragraphs
    paragraphs = []
    with open(para_path) as f:
        for line in f:
            paragraphs.append(json.loads(line))

    # Check for existing checkpoint (partial results)
    checkpoint_path = CLASSIFICATIONS_DIR / f"{stem}.checkpoint.json"
    completed_windows = {}
    start_window = 0

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        completed_windows = {int(k): v for k, v in checkpoint.get("results", {}).items()}
        start_window = checkpoint.get("next_window", 0)
        log.info(f"{stem}: resuming from window {start_window} "
                 f"({len(completed_windows)} completed)")

    # Create sliding windows
    windows = make_windows(paragraphs)
    total_windows = len(windows)
    log.info(f"{stem}: {len(paragraphs)} paragraphs, {total_windows} windows")

    if start_window >= total_windows:
        log.info(f"{stem}: all windows already completed")
    else:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 10)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Process remaining windows in chunks for checkpointing
            CHECKPOINT_EVERY = 50
            pending_windows = list(range(start_window, total_windows))

            for chunk_start in range(0, len(pending_windows), CHECKPOINT_EVERY):
                chunk = pending_windows[chunk_start : chunk_start + CHECKPOINT_EVERY]

                tasks = [
                    classify_batch(
                        session, semaphore, api_url,
                        windows[win_idx],
                        edition_name, year, volume, win_idx,
                    )
                    for win_idx in chunk
                ]

                t0 = time.time()
                results = await asyncio.gather(*tasks)
                elapsed = time.time() - t0

                for win_idx, classifications in results:
                    completed_windows[win_idx] = classifications

                done = chunk[-1] + 1
                rate = len(chunk) / elapsed if elapsed > 0 else 0
                log.info(f"{stem}: {done}/{total_windows} windows "
                         f"({rate:.1f} calls/sec, {elapsed:.1f}s)")

                # Save checkpoint
                with open(checkpoint_path, "w") as f:
                    json.dump({
                        "next_window": done,
                        "results": {str(k): v for k, v in completed_windows.items()},
                    }, f)

    # Merge overlapping results
    merged = merge_classifications(windows, completed_windows)

    # Build full classification list (body_text for unreported paragraphs)
    all_classifications = []
    for p in paragraphs:
        idx = p["index"]
        if idx in merged:
            cls = merged[idx]
            cls_type = cls.get("type", "body_text")
            entry = {"index": idx, "type": cls_type}
            if cls.get("title"):
                entry["title"] = cls["title"]
            if cls.get("keywords"):
                entry["keywords"] = cls["keywords"]
            if cls.get("target"):
                entry["target"] = cls["target"]
        else:
            entry = {"index": idx, "type": "body_text"}
        all_classifications.append(entry)

    # Write output
    with open(output_path, "w") as f:
        for cls in all_classifications:
            f.write(json.dumps(cls, ensure_ascii=False) + "\n")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Stats
    type_counts = {}
    for cls in all_classifications:
        t = cls["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    log.info(f"{stem}: classifications: {type_counts}")

    return output_path


async def run(files: list[Path] | None = None, api_base: str | None = None):
    """Run Phase 2 on all or specified files."""
    ensure_dirs()

    # Resolve API URL
    if api_base:
        api_url = f"{api_base}/chat/completions"
    else:
        api_url = API_URL

    if files is None:
        files = sorted(INPUT_DIR.glob("*.jsonl"))

    for input_path in files:
        stem = input_path.stem
        para_path = PARAGRAPHS_DIR / f"{stem}.paragraphs.jsonl"

        if not para_path.exists():
            log.error(f"No paragraphs file for {stem} — run Phase 1 first")
            continue

        # Skip if already classified (and no checkpoint = fully done)
        cls_path = CLASSIFICATIONS_DIR / f"{stem}.classifications.jsonl"
        chk_path = CLASSIFICATIONS_DIR / f"{stem}.checkpoint.json"
        if cls_path.exists() and not chk_path.exists():
            log.info(f"Skipping {stem} (already classified)")
            continue

        await classify_file(input_path, para_path, api_url)

    log.info(f"Phase 2 complete: {len(files)} files")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(run())
