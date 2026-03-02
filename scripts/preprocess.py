"""Phase 1: Split volume text into paragraphs with offset tracking."""

import json
import logging
from pathlib import Path

from config import INPUT_DIR, PARAGRAPHS_DIR, PREVIEW_LENGTH, ensure_dirs

log = logging.getLogger(__name__)


def split_paragraphs(text: str) -> list[dict]:
    """Split text on \\n\\n boundaries and record offsets."""
    paragraphs = []
    pos = 0
    for i, chunk in enumerate(text.split("\n\n")):
        # Find actual position in original text (accounting for the \n\n delimiters)
        char_start = text.index(chunk, pos)
        char_end = char_start + len(chunk)
        paragraphs.append({
            "index": i,
            "char_start": char_start,
            "char_end": char_end,
            "text": chunk,
            "preview": chunk[:PREVIEW_LENGTH],
        })
        pos = char_end
    return paragraphs


def process_file(input_path: Path) -> Path:
    """Process a single JSONL file into paragraphs."""
    stem = input_path.stem  # e.g. britannica_3rd_1797_vol01_A-ANG
    output_path = PARAGRAPHS_DIR / f"{stem}.paragraphs.jsonl"

    if output_path.exists():
        log.info(f"Skipping {stem} (already exists)")
        return output_path

    with open(input_path) as f:
        data = json.loads(f.readline())

    paragraphs = split_paragraphs(data["text"])

    with open(output_path, "w") as f:
        for p in paragraphs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    log.info(f"{stem}: {len(paragraphs)} paragraphs, "
             f"{len(data['text']):,} chars")
    return output_path


def run(files: list[Path] | None = None):
    """Run Phase 1 on all or specified files."""
    ensure_dirs()

    if files is None:
        files = sorted(INPUT_DIR.glob("*.jsonl"))

    total_paras = 0
    for path in files:
        output = process_file(path)
        # Count paragraphs
        with open(output) as f:
            count = sum(1 for _ in f)
        total_paras += count

    log.info(f"Phase 1 complete: {len(files)} files, {total_paras:,} total paragraphs")
    return total_paras


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
