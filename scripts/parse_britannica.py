#!/usr/bin/env python3
"""
Britannica OCR Article Parser — CLI Orchestrator

Usage:
    python parse_britannica.py --phase lis            # LIS-based article extraction (fast, no GPU)
    python parse_britannica.py --phase lis2           # LIS V2: extraction + cross-edition in one pass
    python parse_britannica.py --phase cross-edition   # Cross-edition headword validation
    python parse_britannica.py --phase 1          # Split all files into paragraphs (legacy LLM pipeline)
    python parse_britannica.py --phase 2          # Classify paragraphs with LLM (legacy)
    python parse_britannica.py --phase 3          # Assemble articles (legacy)
    python parse_britannica.py --phase merge      # Merge fragment articles into parents (legacy)
    python parse_britannica.py --phase all        # Run all LLM phases (legacy: 1 + 2 + 3 + merge)
    python parse_britannica.py --phase verify     # Run quality checks
    python parse_britannica.py --phase dedup      # Deduplicate source OCR files
    python parse_britannica.py --phase compare    # Compare 1st edition vs earlier parser
    python parse_britannica.py --phase audit      # Audit alphabetical order flags
    python parse_britannica.py --phase export     # Export final dataset (JSONL + SQLite)
    python parse_britannica.py --phase site       # Generate static GitHub Pages site
    python parse_britannica.py --phase lis --file britannica_3rd_1797_vol18_STR-ZYM.jsonl
    python parse_britannica.py --phase lis --edition 3rd
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from config import INPUT_DIR, INDEX_1842_PATH, HEADWORD_DICT_PATH, ensure_dirs

log = logging.getLogger("britannica_parser")


def resolve_files(args) -> list[Path] | None:
    """Resolve which input files to process.

    Returns None when no --file or --edition is specified, so that
    lis_parser.run() uses get_canonical_files() instead.
    """
    if args.file:
        path = INPUT_DIR / args.file
        if not path.exists():
            path = INPUT_DIR / f"{args.file}.jsonl"
        if not path.exists():
            log.error(f"File not found: {args.file}")
            sys.exit(1)
        return [path]

    if args.edition:
        # Try both naming conventions (eb_ and britannica_)
        for prefix in ['eb', 'britannica']:
            pattern = f"{prefix}_{args.edition}_*.jsonl"
            files = sorted(INPUT_DIR.glob(pattern))
            files = [f for f in files if '_dup' not in f.stem and '_alt' not in f.stem]
            if files:
                return files
        log.error(f"No files matching edition '{args.edition}'")
        sys.exit(1)

    # All files: return None so lis_parser.run() uses get_canonical_files()
    return None


def run_phase1(files: list[Path]):
    """Run Phase 1: paragraph splitting."""
    import preprocess
    preprocess.run(files)


def run_phase2(files: list[Path], api_base: str | None = None):
    """Run Phase 2: LLM classification."""
    import classify
    asyncio.run(classify.run(files, api_base))


def run_phase3(files: list[Path]):
    """Run Phase 3: article assembly."""
    import assemble
    assemble.run(files)


def run_merge(files: list[Path]):
    """Run Phase 3.5: merge fragment articles into parent treatises."""
    import merge
    merge.run(files)


def run_verify(files: list[Path]):
    """Run verification checks."""
    import verify
    verify.run(files)


def run_dedup(files: list[Path]):
    """Run deduplication of source OCR files."""
    import dedup
    dedup.run(files)


def run_compare(files: list[Path]):
    """Compare 1st edition against earlier hybrid parser."""
    import compare
    compare.run(files)


def run_audit(files: list[Path]):
    """Audit alphabetical order flags."""
    import audit_order
    audit_order.run(files)


def run_export(files: list[Path]):
    """Export final deduplicated dataset."""
    import export
    export.run(files)


def run_lis(files: list[Path], index_path=None):
    """Run LIS-based article extraction (replaces Phases 1-3.5)."""
    import lis_parser
    lis_parser.run(files, index_path)


def run_cross_edition(files: list[Path]):
    """Build cross-edition union index and validate."""
    import cross_edition
    cross_edition.run(files)


def run_site():
    """Generate static GitHub Pages site from export data."""
    import generate_site
    generate_site.run()


def main():
    parser = argparse.ArgumentParser(
        description="Britannica OCR Article Parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", required=True,
        choices=["lis", "lis2", "cross-edition",
                 "1", "2", "3", "merge", "all", "verify",
                 "dedup", "compare", "audit", "export", "site"],
        help="Which phase to run (lis=LIS extraction, lis2=LIS V2 + cross-edition, "
             "cross-edition=validation, "
             "1=split, 2=classify, 3=assemble, merge, all=legacy LLM pipeline, "
             "verify, dedup, compare, audit, export, site)",
    )
    parser.add_argument(
        "--file",
        help="Process a single file (filename in ocr_organized/)",
    )
    parser.add_argument(
        "--edition",
        help="Process all files for an edition (e.g. '3rd', '8th')",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints (Phase 2 only)",
    )
    parser.add_argument(
        "--api-base",
        help="Override LLM API base URL (e.g. http://platogpu002:8000/v1)",
    )
    parser.add_argument(
        "--concurrency", type=int,
        help="Override max concurrent API requests",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Apply overrides
    if args.concurrency:
        import config
        config.MAX_CONCURRENT = args.concurrency

    ensure_dirs()
    files = resolve_files(args)
    if files is not None:
        log.info(f"Processing {len(files)} file(s)")
    else:
        log.info("Processing all canonical files (via manifest)")

    phase = args.phase

    if phase == "lis":
        log.info("=" * 40 + " LIS: Article Extraction " + "=" * 40)
        index_path = INDEX_1842_PATH if INDEX_1842_PATH.exists() else None
        run_lis(files, index_path)

    if phase == "lis2":
        log.info("=" * 40 + " LIS V2: Extraction + Cross-Edition " + "=" * 40)
        index_path = INDEX_1842_PATH if INDEX_1842_PATH.exists() else None
        run_lis(files, index_path)
        log.info("=" * 40 + " CROSS-EDITION: Validation (Pass 2) " + "=" * 40)
        # For cross-edition, we need actual file list
        if files is None:
            files = sorted(INPUT_DIR.glob("*.jsonl"))
            files = [f for f in files if '_dup' not in f.stem and '_alt' not in f.stem]
        run_cross_edition(files)

    if phase == "cross-edition":
        log.info("=" * 40 + " CROSS-EDITION: Validation " + "=" * 40)
        if files is None:
            files = sorted(INPUT_DIR.glob("*.jsonl"))
            files = [f for f in files if '_dup' not in f.stem and '_alt' not in f.stem]
        run_cross_edition(files)

    # For legacy phases that require file lists, resolve if None
    if files is None and phase in ("1", "2", "3", "merge", "all", "verify", "dedup", "compare", "audit", "export"):
        files = sorted(INPUT_DIR.glob("*.jsonl"))
        files = [f for f in files if '_dup' not in f.stem and '_alt' not in f.stem]
        log.info(f"Resolved {len(files)} files (excluding dups/alts)")

    if phase == "1" or phase == "all":
        log.info("=" * 40 + " PHASE 1: Paragraph Splitting " + "=" * 40)
        run_phase1(files)

    if phase == "2" or phase == "all":
        log.info("=" * 40 + " PHASE 2: LLM Classification " + "=" * 40)
        run_phase2(files, args.api_base)

    if phase == "3" or phase == "all":
        log.info("=" * 40 + " PHASE 3: Article Assembly " + "=" * 40)
        run_phase3(files)

    if phase == "merge" or phase == "all":
        log.info("=" * 40 + " PHASE 3.5: Merge Fragments " + "=" * 40)
        run_merge(files)

    if phase == "verify":
        log.info("=" * 40 + " VERIFICATION " + "=" * 40)
        run_verify(files)

    if phase == "all":
        log.info("=" * 40 + " VERIFICATION " + "=" * 40)
        run_verify(files)

    if phase == "dedup":
        log.info("=" * 40 + " DEDUP: Source File Deduplication " + "=" * 40)
        run_dedup(files)

    if phase == "compare":
        log.info("=" * 40 + " COMPARE: vs Earlier Parser " + "=" * 40)
        run_compare(files)

    if phase == "audit":
        log.info("=" * 40 + " AUDIT: Alphabetical Order Flags " + "=" * 40)
        run_audit(files)

    if phase == "export":
        log.info("=" * 40 + " EXPORT: Final Dataset " + "=" * 40)
        run_export(files)

    if phase == "site":
        log.info("=" * 40 + " SITE: Static GitHub Pages " + "=" * 40)
        run_site()

    log.info("Done.")


if __name__ == "__main__":
    main()
