#!/usr/bin/env python3
"""
Build structured co-occurrence data for visualization.

Generates two JSON datasets:
1. commodity_by_place.json — for each place, what commodities appear nearby over time
2. place_by_commodity.json — for each commodity, what places appear nearby over time

Usage:
    python graphrag/build_colocation_data.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "data" / "export"
NER_DIR = REPO_ROOT / "data" / "ner"
OUT_DIR = REPO_ROOT / "data" / "ner"

EDITION_YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]
WINDOW = 150

COMMODITIES = [
    "sugar", "cotton", "indigo", "opium", "tea", "silk", "wool",
    "gold", "silver", "copper", "iron", "slaves", "negroes",
    "tobacco", "coffee", "rice", "ginger", "rhubarb", "cinchona",
    "ivory", "timber", "furs", "rubber", "diamond", "coal",
]

PLACES = [
    "China", "India", "Bengal", "Jamaica", "Barbados", "Brazil",
    "Canada", "Mexico", "Peru", "Egypt", "Turkey", "Persia",
    "Java", "Sumatra", "West Indies", "East Indies", "Africa",
    "France", "England", "Spain", "Holland", "Germany",
    "United States", "Manchester", "Glasgow", "Lancashire",
    "Gold Coast", "Cape Colony", "Demerara", "Calcutta",
    "Bombay", "Madras", "Canton", "Havannah", "Grenada",
    "Antigua", "Nevis", "Trinidad", "St Domingo", "Cuba",
    "Carolina", "Virginia", "Guatemala", "Paraguay",
    "Hindustan", "Ceylon", "Malabar", "Arabia",
]

TOPONYM_SKIP = {
    "W. Long.", "W. Long", "N. Lat.", "N. Lat", "E. Long.", "E. Long",
    "CANTARO", "river", "east", "west", "north", "south",
    "Bleaching", "Ammonia", "Common", "Hat", "orange-peel",
    "Female", "Lead", "Staple", "Radish", "Oxalis", "Cows",
}


def find_all_occurrences(text_lower, term):
    positions = []
    start = 0
    term_lower = term.lower()
    while True:
        idx = text_lower.find(term_lower, start)
        if idx == -1:
            break
        before_ok = idx == 0 or not text_lower[idx - 1].isalpha()
        after_end = idx + len(term_lower)
        after_ok = after_end >= len(text_lower) or not text_lower[after_end].isalpha()
        if before_ok and after_ok:
            positions.append(idx)
        start = idx + 1
    return positions


def char_to_word_index(text):
    mapping = [0] * len(text)
    word_idx = 0
    in_word = False
    for i, ch in enumerate(text):
        if ch.isspace():
            in_word = False
        else:
            if not in_word:
                word_idx += 1
                in_word = True
        mapping[i] = word_idx
    return mapping


def build_data():
    # Result: {year: {commodity: {place: count}}} and {year: {place: {commodity: count}}}
    by_commodity = defaultdict(lambda: defaultdict(Counter))
    by_place = defaultdict(lambda: defaultdict(Counter))

    for year in EDITION_YEARS:
        print(f"Processing {year}...", end=" ", flush=True)

        ner_files = [f for f in NER_DIR.glob(f"eb_*_{year}.entities.jsonl")
                     if "_v" not in f.stem]
        export_files = list(EXPORT_DIR.glob(f"eb_*_{year}.jsonl"))
        if not ner_files or not export_files:
            print("skipped")
            continue

        # Build article lookups from NER
        article_commodities = {}
        article_toponyms = {}
        with open(ner_files[0]) as f:
            for line in f:
                rec = json.loads(line)
                aid = rec["article_id"]
                article_commodities[aid] = [
                    e["text"] for e in rec["entities"] if e["type"] == "COMMODITY"
                ]
                article_toponyms[aid] = [
                    e["text"] for e in rec["entities"]
                    if e["type"] == "TOPONYM" and e["text"] not in TOPONYM_SKIP
                ]

        # Process each article
        n_articles = 0
        with open(export_files[0]) as f:
            for line in f:
                art = json.loads(line)
                aid = art["article_id"]
                text = art.get("text", "")
                if not text:
                    continue

                commodities = article_commodities.get(aid, [])
                toponyms = article_toponyms.get(aid, [])
                if not commodities or not toponyms:
                    continue

                text_lower = text.lower()
                word_map = char_to_word_index(text)

                # Pre-compute positions for all commodities and toponyms in this article
                commodity_positions = {}
                for c in commodities:
                    positions = find_all_occurrences(text_lower, c.lower())
                    if positions:
                        commodity_positions[c] = [word_map[p] for p in positions]

                toponym_positions = {}
                for t in toponyms:
                    positions = find_all_occurrences(text_lower, t.lower())
                    if positions:
                        toponym_positions[t] = [word_map[p] for p in positions]

                if not commodity_positions or not toponym_positions:
                    continue

                n_articles += 1

                # Check all commodity-toponym pairs for proximity
                for commodity, c_words in commodity_positions.items():
                    c_lower = commodity.lower()
                    for toponym, t_words in toponym_positions.items():
                        # Check if any pair is within window
                        nearby = False
                        for cw in c_words:
                            for tw in t_words:
                                if 0 < abs(cw - tw) <= WINDOW:
                                    nearby = True
                                    break
                            if nearby:
                                break
                        if nearby:
                            by_commodity[year][c_lower][toponym] += 1
                            by_place[year][toponym][c_lower] += 1

        print(f"{n_articles} articles")

    return by_commodity, by_place


def save_data(by_commodity, by_place):
    # Convert to serializable format
    commodity_data = {}
    for year in EDITION_YEARS:
        commodity_data[year] = {}
        for commodity, places in by_commodity[year].items():
            commodity_data[year][commodity] = dict(places.most_common(50))

    place_data = {}
    for year in EDITION_YEARS:
        place_data[year] = {}
        for place, commodities in by_place[year].items():
            place_data[year][place] = dict(commodities.most_common(50))

    out1 = OUT_DIR / "colocation_by_commodity.json"
    out2 = OUT_DIR / "colocation_by_place.json"

    with open(out1, "w") as f:
        json.dump(commodity_data, f, indent=2, ensure_ascii=False)
    with open(out2, "w") as f:
        json.dump(place_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    by_commodity, by_place = build_data()
    save_data(by_commodity, by_place)
