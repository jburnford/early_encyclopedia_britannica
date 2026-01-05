#!/usr/bin/env python3
"""
LLM Article Corrector - Main Orchestration Script

Generates prompts for LLM review of flagged articles and processes responses.

Usage:
    # Generate prompt for next unprocessed batch
    python3 llm_article_corrector.py --prompt

    # Generate prompt for specific edition/batch
    python3 llm_article_corrector.py --prompt --edition 1771 --batch 1

    # Record decisions from LLM response (interactive)
    python3 llm_article_corrector.py --record

    # Show current progress
    python3 llm_article_corrector.py --status
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Directories
SCRIPT_DIR = Path(__file__).parent
STATE_DIR = SCRIPT_DIR / "state"
PROMPTS_DIR = SCRIPT_DIR / "prompts"
CORRECTIONS_DIR = SCRIPT_DIR / "corrections"

# Ensure directories exist
STATE_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_progress() -> dict:
    """Load current progress state."""
    progress_file = STATE_DIR / "progress.json"
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "total_flagged": 0,
        "total_batches": 0,
        "processed": 0,
        "current_edition": None,
        "current_batch": 0,
        "by_edition": {},
        "decisions": {"merge": 0, "keep_separate": 0, "delete": 0}
    }


def save_progress(progress: dict):
    """Save progress state."""
    progress_file = STATE_DIR / "progress.json"
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


def load_decisions() -> list[dict]:
    """Load all recorded decisions."""
    decisions_file = CORRECTIONS_DIR / "decisions.json"
    if decisions_file.exists():
        with open(decisions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_decisions(decisions: list[dict]):
    """Save decisions."""
    decisions_file = CORRECTIONS_DIR / "decisions.json"
    with open(decisions_file, 'w', encoding='utf-8') as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)


def find_next_batch(progress: dict) -> Optional[tuple[int, int]]:
    """Find the next unprocessed batch."""
    processed_batches = set()

    # Load all decisions to find which batches are done
    decisions = load_decisions()
    for d in decisions:
        key = f"{d.get('edition_year')}_{d.get('batch_num')}"
        processed_batches.add(key)

    # Find batch files and check which are unprocessed
    for batch_file in sorted(STATE_DIR.glob("batch_*.json")):
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)

        edition = batch_data.get('edition_year')
        batch_num = batch_data.get('batch_num')
        key = f"{edition}_{batch_num}"

        if key not in processed_batches:
            return (edition, batch_num)

    return None


def load_batch(edition_year: int, batch_num: int) -> Optional[dict]:
    """Load a specific batch file."""
    batch_file = STATE_DIR / f"batch_{edition_year}_{batch_num:03d}.json"
    if batch_file.exists():
        with open(batch_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def generate_prompt(batch_data: dict) -> str:
    """Generate LLM prompt for a batch."""
    edition = batch_data.get('edition_year')
    batch_num = batch_data.get('batch_num')
    total_batches = batch_data.get('total_batches')
    articles = batch_data.get('articles', [])

    # Count issue types in this batch
    issue_counts = {}
    for article in articles:
        issue_type = article['flagged'].get('primary_issue', 'unknown')
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

    prompt = f"""# Encyclopedia Britannica Article Merge Review

**Edition**: {edition} | **Batch**: {batch_num} of {total_batches} | **Articles**: {len(articles)}
**Issues in this batch**: {', '.join(f'{k}: {v}' for k, v in issue_counts.items())}

---

## Instructions

Review each flagged article and decide:

- **MERGE**: Flagged article is a mis-parsed section heading that should merge into parent
- **KEEP**: Flagged article is a valid standalone encyclopedia entry
- **DELETE**: Structural marker or OCR garbage (rare)

**Key signals for MERGE**:
1. Flagged headword is generic ("GENERAL OBSERVATIONS", "DISEASES OF THE FEET")
2. Text continues the topic of parent article
3. First letter doesn't match surrounding articles (e.g., 'B' surrounded by 'C')

**Key signals for KEEP**:
1. Flagged article is a complete, self-contained topic
2. Has proper encyclopedia structure (definition, explanation)
3. Could stand alone even if alphabetically misplaced

---

"""

    for i, article in enumerate(articles, 1):
        flagged = article['flagged']
        # Handle both old (parent_candidate) and new (parent_candidates) format
        parents = article.get('parent_candidates', {})
        if not parents and article.get('parent_candidate'):
            # Legacy format
            parents = {'page_adjacent': article.get('parent_candidate'), 'semantic': None}

        page_parent = parents.get('page_adjacent')
        semantic_parent = parents.get('semantic')

        headword = flagged.get('headword', 'UNKNOWN')
        page = flagged.get('start_page', '?')
        end_page = flagged.get('end_page', page)
        word_count = flagged.get('word_count', 0)
        surrounding = flagged.get('surrounding_letter', '?')
        primary_issue = flagged.get('primary_issue', 'unknown')
        text_preview = flagged.get('text_preview', '')

        # Get first letter
        first_letter = ''
        for c in headword:
            if c.isalpha():
                first_letter = c.upper()
                break

        prompt += f"""### {i}. {headword}

**Page**: {page}-{end_page} | **Words**: {word_count} | **Issue**: {primary_issue}
**First letter**: '{first_letter}' surrounded by '{surrounding}' articles

"""

        # Show semantic parent first if available (e.g., "BLACK CHALK" -> "CHALK")
        if semantic_parent:
            sem_hw = semantic_parent.get('headword', 'UNKNOWN')
            sem_page = semantic_parent.get('end_page', semantic_parent.get('start_page', '?'))
            sem_text = semantic_parent.get('text_end', '')

            prompt += f"""**Semantic parent** (headword match): **{sem_hw}** (p.{sem_page})
> ...{sem_text}

"""

        # Show page-adjacent parent
        if page_parent:
            adj_hw = page_parent.get('headword', 'UNKNOWN')
            adj_page = page_parent.get('end_page', page_parent.get('start_page', '?'))
            adj_text = page_parent.get('text_end', '')

            prompt += f"""**Page-adjacent parent** (previous article): {adj_hw} (ends p.{adj_page})
> ...{adj_text}

"""

        prompt += f"""**Flagged article text**:
> {text_preview}...

"""

        # Build decision options based on available parents
        merge_options = []
        if semantic_parent:
            merge_options.append(f"MERGE into **{semantic_parent.get('headword')}** (semantic)")
        if page_parent and (not semantic_parent or page_parent.get('headword') != semantic_parent.get('headword')):
            merge_options.append(f"MERGE into {page_parent.get('headword')} (page-adjacent)")

        if merge_options:
            decision_line = " / ".join([f"[ ] {opt}" for opt in merge_options])
            decision_line += " / [ ] KEEP / [ ] DELETE"
        else:
            decision_line = "[ ] KEEP / [ ] DELETE"

        prompt += f"""**Decision**: {decision_line}
**Reasoning**:

---

"""

    prompt += """
## Summary

After reviewing all articles, provide a summary in this format:

```json
[
  {"num": 1, "decision": "MERGE", "into": "PARENT_HEADWORD", "reasoning": "Brief reason"},
  {"num": 2, "decision": "KEEP", "reasoning": "Brief reason"},
  ...
]
```
"""

    return prompt


def show_status():
    """Show current progress status."""
    progress = load_progress()
    decisions = load_decisions()

    print("\n" + "=" * 60)
    print("LLM Article Correction Progress")
    print("=" * 60)

    print(f"\nTotal flagged articles: {progress.get('total_flagged', 0)}")
    print(f"Total batches: {progress.get('total_batches', 0)}")
    print(f"Decisions recorded: {len(decisions)}")

    # Count decisions by type
    decision_counts = {"merge": 0, "keep_separate": 0, "delete": 0}
    for d in decisions:
        dec = d.get('decision', '').lower()
        if 'merge' in dec:
            decision_counts['merge'] += 1
        elif 'keep' in dec:
            decision_counts['keep_separate'] += 1
        elif 'delete' in dec:
            decision_counts['delete'] += 1

    print(f"\nDecision breakdown:")
    print(f"  Merge: {decision_counts['merge']}")
    print(f"  Keep: {decision_counts['keep_separate']}")
    print(f"  Delete: {decision_counts['delete']}")

    # Show by-edition progress
    if progress.get('by_edition'):
        print("\nBy edition:")
        for edition, stats in progress.get('by_edition', {}).items():
            batch_decisions = [d for d in decisions if str(d.get('edition_year')) == str(edition)]
            print(f"  {edition}: {len(batch_decisions)}/{stats.get('flagged', 0)} articles")

    # Find next batch
    next_batch = find_next_batch(progress)
    if next_batch:
        print(f"\nNext batch: {next_batch[0]} edition, batch {next_batch[1]}")
    else:
        print("\nAll batches processed!")

    print()


def generate_prompt_cmd(edition: Optional[int], batch_num: Optional[int]):
    """Generate and save prompt for a batch."""
    progress = load_progress()

    if edition and batch_num:
        target = (edition, batch_num)
    else:
        target = find_next_batch(progress)

    if not target:
        print("No unprocessed batches found!")
        return

    edition_year, batch_num = target

    batch_data = load_batch(edition_year, batch_num)
    if not batch_data:
        print(f"Could not load batch {edition_year}_{batch_num}")
        return

    prompt = generate_prompt(batch_data)

    # Save to file
    prompt_file = PROMPTS_DIR / f"prompt_{edition_year}_{batch_num:03d}.md"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)

    print(f"\nGenerated prompt: {prompt_file}")
    print(f"Edition: {edition_year} | Batch: {batch_num}")
    print(f"Articles: {len(batch_data.get('articles', []))}")
    print(f"\nCopy the prompt content and paste to Claude for review.")
    print(f"\nAfter getting responses, run:")
    print(f"  python3 llm_article_corrector.py --record --edition {edition_year} --batch {batch_num}")


def record_decisions_cmd(edition: int, batch_num: int):
    """Record decisions from LLM response."""
    batch_data = load_batch(edition, batch_num)
    if not batch_data:
        print(f"Could not load batch {edition}_{batch_num}")
        return

    articles = batch_data.get('articles', [])
    decisions = load_decisions()

    print(f"\nRecording decisions for {edition} edition, batch {batch_num}")
    print(f"Articles in batch: {len(articles)}")
    print("\nFor each article, enter: m (merge), k (keep), d (delete)")
    print("Or paste JSON array of decisions.\n")

    # Check for JSON input
    print("Enter decisions (JSON array or individual):")
    first_input = input().strip()

    if first_input.startswith('['):
        # JSON input
        try:
            json_decisions = json.loads(first_input)
            for jd in json_decisions:
                num = jd.get('num', 0) - 1
                if 0 <= num < len(articles):
                    article = articles[num]
                    decision_type = jd.get('decision', '').upper()
                    merge_into = jd.get('into')
                    reasoning = jd.get('reasoning', '')

                    decisions.append({
                        "article_id": article['flagged'].get('article_id'),
                        "headword": article['flagged'].get('headword'),
                        "edition_year": edition,
                        "batch_num": batch_num,
                        "decision": decision_type,
                        "merge_into": merge_into,
                        "reasoning": reasoning,
                        "recorded_at": datetime.now().isoformat()
                    })

            save_decisions(decisions)
            print(f"\nRecorded {len(json_decisions)} decisions.")
            return

        except json.JSONDecodeError:
            print("Invalid JSON. Switching to interactive mode.\n")

    # Interactive mode
    for i, article in enumerate(articles):
        flagged = article['flagged']
        parent = article.get('parent_candidate')

        headword = flagged.get('headword', 'UNKNOWN')
        parent_hw = parent.get('headword', 'N/A') if parent else 'N/A'

        print(f"\n{i+1}. {headword}")
        print(f"   Parent candidate: {parent_hw}")

        while True:
            choice = input("   Decision [m/k/d]: ").strip().lower()
            if choice in ['m', 'k', 'd']:
                break
            print("   Invalid. Enter m (merge), k (keep), or d (delete)")

        decision_type = {'m': 'MERGE', 'k': 'KEEP', 'd': 'DELETE'}[choice]

        reasoning = ""
        if choice == 'm':
            reasoning = input("   Reasoning (optional): ").strip()
            merge_into = parent_hw if parent else None
        else:
            merge_into = None
            if choice == 'k':
                reasoning = input("   Reasoning (optional): ").strip()

        decisions.append({
            "article_id": flagged.get('article_id'),
            "headword": headword,
            "edition_year": edition,
            "batch_num": batch_num,
            "decision": decision_type,
            "merge_into": merge_into,
            "reasoning": reasoning,
            "recorded_at": datetime.now().isoformat()
        })

    save_decisions(decisions)
    print(f"\nRecorded {len(articles)} decisions.")

    # Update progress
    progress = load_progress()
    progress['processed'] += len(articles)
    save_progress(progress)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Article Corrector - Generate prompts and record decisions"
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Generate prompt for next/specified batch"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record decisions from LLM response"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current progress"
    )
    parser.add_argument(
        "--edition",
        type=int,
        help="Specific edition year"
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Specific batch number"
    )

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.prompt:
        generate_prompt_cmd(args.edition, args.batch)
    elif args.record:
        if not args.edition or not args.batch:
            print("Error: --record requires --edition and --batch")
            sys.exit(1)
        record_decisions_cmd(args.edition, args.batch)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
