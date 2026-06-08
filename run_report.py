"""Génère les résultats réels sans Jupyter.

Interroge la gateway pour chaque événement de data/events.json, construit le tableau des scores
et écrit tous les résultats dans results/ (scores.csv, correlations.csv, correlations.png).
Prérequis : les services tournent (python run_all.py) et data/prepare_data.py a été exécuté.
"""
import json
from pathlib import Path

import httpx
import pandas as pd

from analysis import correlation_table, save_results

HERE = Path(__file__).resolve().parent
GATEWAY = "http://localhost:8000/analyze"


def main():
    events = json.loads((HERE / "data" / "events.json").read_text())
    rows = []
    for ev in events:
        print(f"→ {ev['id']} ...", flush=True)
        res = httpx.post(GATEWAY, json={"event_id": ev["id"]}, timeout=900).json()
        by = {c["channel"]: c["score"] for c in res["channels"]}
        signal_path = HERE / "data" / "processed" / ev["id"] / "market_signal.json"
        signal = json.loads(signal_path.read_text())["signal_pct"] if signal_path.exists() else None
        rows.append({
            "event": ev["id"],
            "nlp": by.get("nlp"),
            "audio": by.get("audio"),
            "vision": by.get("vision"),
            "fusion": res["combined_score"],
            "market": signal,
        })
    df = pd.DataFrame(rows)
    out = save_results(df)
    print("\n=== scores ===")
    print(df.to_string(index=False))
    print("\n=== corrélation avec le marché ===")
    print(correlation_table(df).to_string())
    print(f"\nRésultats écrits dans : {out}")


if __name__ == "__main__":
    main()
