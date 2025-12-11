import json
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

LOG_DIR = Path("intent_logs")

def load_logs(last_n_days=30):
    rows = []
    cutoff = datetime.utcnow() - timedelta(days=last_n_days)

    for path in sorted(LOG_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ts = datetime.fromisoformat(rec["timestamp"].replace("Z", ""))
                if ts >= cutoff:
                    rows.append(rec)

    return pd.DataFrame(rows)


df = load_logs(last_n_days=90)

if df.empty:
    print("No logs found in the last 90 days.")
else:
    # Explode intents into separate rows
    intents_df = df.explode("intents", ignore_index=True)
    intents_df["intent_label"] = intents_df["intents"].apply(
        lambda x: x.get("label") if isinstance(x, dict) else None
    )

    # Aggregate counts per intent
    intent_counts = (
        intents_df
        .groupby("intent_label", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Pretty display
    print("=== Intent Counts (Table View) ===")
    display(intent_counts.style.background_gradient(cmap='Blues').set_properties(**{'text-align': 'center'}))

    # Save outputs
    intent_counts.to_csv("intent_counts.csv", index=False, encoding="utf-8")
    intent_counts.to_csv("intent_counts.txt", index=False, sep="\t", encoding="utf-8")

    print("\n Saved files:")
    print("intent_counts.csv")
    print("intent_counts.txt")