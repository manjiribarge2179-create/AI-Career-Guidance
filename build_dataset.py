"""
Converts AI_Career_Guidance_Dataset_100.csv to data/career_dataset.json
Run once: python build_dataset.py
"""
import csv
import json
import os

CSV_PATH = "AI_Career_Guidance_Dataset_100.csv"
OUT_PATH = "data/career_dataset.json"

os.makedirs("data", exist_ok=True)

entries = []
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row["question"].strip()
        a = row["answer"].strip()
        t = row["topic"].strip()
        lvl = row["student_level"].strip()
        if q and a:
            entries.append({
                "input": q,
                "response": a,
                "topic": t,
                "level": lvl
            })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)

print(f"[build_dataset] {len(entries)} entries written to {OUT_PATH}")
