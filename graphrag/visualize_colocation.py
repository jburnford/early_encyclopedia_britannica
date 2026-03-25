#!/usr/bin/env python3
"""
Build interactive HTML visualizations of commodity-place co-occurrence.

Generates:
1. Heatmaps showing how commodity associations change over time for key places
2. Bump charts showing place rankings for key commodities over time
3. Place profile small multiples

Usage:
    python graphrag/visualize_colocation.py
"""

import json
import numpy as np
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parent.parent
NER_DIR = REPO_ROOT / "data" / "ner"
VIZ_DIR = REPO_ROOT / "data" / "viz"

EDITION_YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]
YEAR_LABELS = ["1771<br>1st", "1778<br>2nd", "1797<br>3rd", "1810<br>4th",
               "1815<br>5th", "1823<br>6th", "1842<br>7th", "1860<br>8th"]


def load_data():
    with open(NER_DIR / "colocation_by_commodity.json") as f:
        by_commodity = json.load(f)
    with open(NER_DIR / "colocation_by_place.json") as f:
        by_place = json.load(f)
    return by_commodity, by_place


def build_place_commodity_heatmap(by_place, places, title, filename,
                                  top_n=15):
    """Heatmap: for each place, top commodities over time."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=len(places), cols=1,
        subplot_titles=[f"{p}" for p in places],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    for pi, place in enumerate(places):
        # Collect all commodities for this place across editions
        all_commodities = set()
        for year in EDITION_YEARS:
            yr_data = by_place.get(str(year), {})
            place_data = yr_data.get(place, {})
            for c, count in sorted(place_data.items(), key=lambda x: -x[1])[:top_n]:
                all_commodities.add(c)

        if not all_commodities:
            continue

        # Rank commodities by total mentions across all editions
        totals = {}
        for c in all_commodities:
            totals[c] = sum(
                by_place.get(str(y), {}).get(place, {}).get(c, 0)
                for y in EDITION_YEARS
            )
        top_commodities = sorted(totals, key=lambda x: -totals[x])[:top_n]

        # Build matrix
        matrix = []
        for c in top_commodities:
            row = []
            for year in EDITION_YEARS:
                val = by_place.get(str(year), {}).get(place, {}).get(c, 0)
                row.append(val)
            matrix.append(row)

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=YEAR_LABELS,
                y=top_commodities,
                colorscale="YlOrRd",
                showscale=(pi == 0),
                colorbar=dict(title="Mentions", len=0.15, y=0.95),
                hovertemplate="%{y} × %{x}: %{z} mentions<extra></extra>",
            ),
            row=pi + 1, col=1,
        )

    height = max(400, len(places) * 350)
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=height,
        width=900,
        template="plotly_white",
    )

    outpath = VIZ_DIR / filename
    fig.write_html(str(outpath), include_plotlyjs="cdn")
    print(f"  Saved: {outpath}")


def build_commodity_bump_chart(by_commodity, commodity, filename,
                                top_n=12):
    """Bump chart: rank of places for a commodity over time."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all places that ever appear in top_n for this commodity
    all_places = set()
    for year in EDITION_YEARS:
        yr_data = by_commodity.get(str(year), {})
        com_data = yr_data.get(commodity, {})
        for place, _ in sorted(com_data.items(), key=lambda x: -x[1])[:top_n]:
            all_places.add(place)

    if not all_places:
        print(f"  No data for {commodity}")
        return

    # Build traces for each place
    fig = go.Figure()

    # Calculate rankings per year
    rankings = {}  # place -> [rank_per_year]
    values = {}    # place -> [value_per_year]
    for place in all_places:
        rankings[place] = []
        values[place] = []
        for year in EDITION_YEARS:
            yr_data = by_commodity.get(str(year), {})
            com_data = yr_data.get(commodity, {})
            val = com_data.get(place, 0)
            values[place].append(val)
            # Rank
            sorted_places = sorted(com_data.items(), key=lambda x: -x[1])
            rank = None
            for i, (p, _) in enumerate(sorted_places):
                if p == place:
                    rank = i + 1
                    break
            rankings[place].append(rank)

    # Sort by best-ever rank
    best_rank = {p: min((r for r in rankings[p] if r is not None), default=999)
                 for p in all_places}
    sorted_places = sorted(all_places, key=lambda p: best_rank[p])[:top_n]

    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    for i, place in enumerate(sorted_places):
        ranks = rankings[place]
        vals = values[place]

        # For display, invert rank (higher = better) and handle None
        y_vals = []
        x_vals = []
        hover_texts = []
        for j, (rank, val) in enumerate(zip(ranks, vals)):
            if rank is not None and rank <= top_n * 2:
                y_vals.append(top_n + 1 - rank)
                x_vals.append(EDITION_YEARS[j])
                hover_texts.append(
                    f"{place}<br>{EDITION_YEARS[j]}: rank #{rank} ({val} mentions)"
                )

        if not x_vals:
            continue

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers+text",
            name=place,
            text=[place if j == len(x_vals) - 1 else "" for j in range(len(x_vals))],
            textposition="middle right",
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=8),
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{commodity.title()}</b> — Place Rankings Over Time<br>"
                 f"<sub>Based on windowed co-occurrence (150 words) in Encyclopedia Britannica</sub>",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Edition Year",
            tickvals=EDITION_YEARS,
            ticktext=[f"{y}" for y in EDITION_YEARS],
        ),
        yaxis=dict(
            title="Rank",
            tickvals=list(range(1, top_n + 1)),
            ticktext=[f"#{top_n + 1 - i}" for i in range(1, top_n + 1)],
            range=[0, top_n + 1.5],
        ),
        height=600,
        width=1000,
        template="plotly_white",
        legend=dict(x=1.05, y=1),
        margin=dict(r=150),
    )

    outpath = VIZ_DIR / filename
    fig.write_html(str(outpath), include_plotlyjs="cdn")
    print(f"  Saved: {outpath}")


def build_place_timeline(by_place, place, filename, top_n=10):
    """Stacked area chart: commodity mix for a place over time."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all commodities for this place
    all_commodities = set()
    for year in EDITION_YEARS:
        yr_data = by_place.get(str(year), {})
        place_data = yr_data.get(place, {})
        for c in place_data:
            all_commodities.add(c)

    if not all_commodities:
        print(f"  No data for {place}")
        return

    # Rank by total
    totals = {c: sum(by_place.get(str(y), {}).get(place, {}).get(c, 0)
                     for y in EDITION_YEARS) for c in all_commodities}
    top_commodities = sorted(totals, key=lambda x: -totals[x])[:top_n]

    fig = go.Figure()
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Dark2

    for i, commodity in enumerate(reversed(top_commodities)):
        vals = [by_place.get(str(y), {}).get(place, {}).get(commodity, 0)
                for y in EDITION_YEARS]
        fig.add_trace(go.Bar(
            x=YEAR_LABELS,
            y=vals,
            name=commodity,
            marker_color=colors[i % len(colors)],
            hovertemplate=f"{commodity}: %{{y}} mentions<extra>{place}</extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{place}</b> — Commodity Associations Over Time<br>"
                 f"<sub>Top {top_n} commodities by windowed co-occurrence in Encyclopedia Britannica</sub>",
            font=dict(size=18),
        ),
        barmode="stack",
        xaxis=dict(title="Edition"),
        yaxis=dict(title="Co-occurrence count (within 150 words)"),
        height=500,
        width=900,
        template="plotly_white",
        legend=dict(traceorder="reversed"),
    )

    outpath = VIZ_DIR / filename
    fig.write_html(str(outpath), include_plotlyjs="cdn")
    print(f"  Saved: {outpath}")


def build_dashboard(by_commodity, by_place):
    """Build a single dashboard HTML with all key visualizations."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Key commodities for bump charts
    key_commodities = ["sugar", "cotton", "opium", "indigo", "tea", "slaves",
                       "wool", "gold", "coffee", "ivory", "tobacco", "silk"]

    # Key places for timelines
    key_places = ["China", "India", "Jamaica", "Brazil", "Canada",
                  "Gold Coast", "Egypt", "Turkey", "Mexico", "West Indies"]

    print("Building commodity bump charts...")
    for commodity in key_commodities:
        build_commodity_bump_chart(
            by_commodity, commodity,
            f"bump_{commodity}.html",
        )

    print("\nBuilding place timelines...")
    for place in key_places:
        build_place_timeline(
            by_place, place,
            f"place_{place.lower().replace(' ', '_')}.html",
        )

    print("\nBuilding heatmaps...")
    build_place_commodity_heatmap(
        by_place,
        ["China", "India", "Jamaica", "Brazil", "Gold Coast", "Canada"],
        "Commodity Associations by Place — Encyclopedia Britannica (1771-1860)",
        "heatmap_places.html",
    )

    # Build index page
    index = """<!DOCTYPE html>
<html><head><title>Encyclopedia Britannica Commodity-Place Co-occurrence</title>
<style>
body { font-family: Georgia, serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
h1 { border-bottom: 2px solid #8B4513; padding-bottom: 10px; }
h2 { color: #8B4513; margin-top: 30px; }
a { color: #8B4513; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.grid a { padding: 8px 12px; background: #f5f0eb; border-radius: 4px; text-decoration: none; }
.grid a:hover { background: #e8ddd3; }
p.method { font-size: 0.9em; color: #666; line-height: 1.6; }
</style></head><body>
<h1>Commodity-Place Co-occurrence in the Encyclopedia Britannica</h1>
<p class="method">
Analysis of 1.15 million named entities extracted from 8 editions (1771-1860) using
<a href="https://github.com/jacobpol/earlymodernner">EarlyModernNER</a>.
Co-occurrence is measured within a 150-word window in the article text,
providing paragraph-level locality rather than article-level association.
</p>

<h2>Commodity Geography Over Time</h2>
<p>How does the geography of each commodity shift across editions?</p>
<div class="grid">
"""
    for c in key_commodities:
        index += f'<a href="bump_{c}.html">{c.title()}</a>\n'

    index += """</div>

<h2>Place Commodity Profiles</h2>
<p>What commodities does the Britannica associate with each place?</p>
<div class="grid">
"""
    for p in key_places:
        slug = p.lower().replace(" ", "_")
        index += f'<a href="place_{slug}.html">{p}</a>\n'

    index += """</div>

<h2>Heatmap Overview</h2>
<p><a href="heatmap_places.html">Combined heatmap</a> — Six key places side by side</p>

</body></html>
"""
    index_path = VIZ_DIR / "index.html"
    with open(index_path, "w") as f:
        f.write(index)
    print(f"\nIndex: {index_path}")


if __name__ == "__main__":
    by_commodity, by_place = load_data()
    build_dashboard(by_commodity, by_place)
