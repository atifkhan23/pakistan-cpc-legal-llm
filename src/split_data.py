"""
Validate the dataset and create the stratified 80/10/10 split (seed 42).

Extracted from notebooks/CPC_PKL_full_run.ipynb (Cell 2).
Paths assume the Kaggle layout (/kaggle/working); change them for local runs.
"""
import json, random, os
from collections import Counter

DATA_PATH = "final_legal_dataset.json"   # path to the dataset JSON
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ===== CELL 2: Validate + stratified 80/10/10 split =====
import json, random
from collections import Counter

SEED = 42
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data, dict):
    data = next(v for v in data.values() if isinstance(v, list))

bad = [i for i, x in enumerate(data)
       if not {"instruction", "output", "type"} <= set(x) or not x["instruction"].strip() or not x["output"].strip()]
print(f"Total examples: {len(data)} | Invalid rows: {len(bad)}")
assert not bad, f"Invalid rows: {bad[:10]}"
print("Categories:", dict(Counter(x["type"] for x in data)))

random.seed(SEED)
by_cat = {}
for i, x in enumerate(data):
    x["id"] = i
    by_cat.setdefault(x["type"], []).append(x)

train, val, test = [], [], []
for cat, items in by_cat.items():
    random.shuffle(items)
    k = round(len(items) * 0.10)
    test += items[:k]; val += items[k:2 * k]; train += items[2 * k:]
for s in (train, val, test):
    random.shuffle(s)

def save(rows, name):
    with open(f"{DATA_DIR}/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "question": r["instruction"].strip(),
                                "answer": r["output"].strip(), "category": r["type"]},
                               ensure_ascii=False) + "\n")
save(train, "train"); save(val, "val"); save(test, "test")

print(f"\ntrain {len(train)} | val {len(val)} | test {len(test)}")
print(f"{'Category':<14}{'train':>7}{'val':>6}{'test':>6}")
for cat in by_cat:
    print(f"{cat:<14}{sum(r['type']==cat for r in train):>7}{sum(r['type']==cat for r in val):>6}{sum(r['type']==cat for r in test):>6}")
