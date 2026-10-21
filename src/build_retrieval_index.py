"""
Build the hybrid retrieval index (BGE-base-en-v1.5 + BM25 + provision boost)
over the train+val QA pairs and retrieve the top-3 entries for each test question.

Extracted from notebooks/CPC_PKL_full_run.ipynb (Cell 10).
Paths assume the Kaggle layout (/kaggle/working); change them for local runs.
"""
# ===== CELL 10: RAG retrieval index (hybrid: dense BGE + BM25 + provision match) =====
# Knowledge base = train + val answers only (never the test answers -> no leakage)
import subprocess, sys, importlib.util
for pkg, pipname in [("rank_bm25", "rank_bm25"), ("sentence_transformers", "sentence-transformers==3.4.1")]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", pipname], check=True)

import json, os, re, numpy as np, torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

WORK, DATA_DIR = "/kaggle/working", "/kaggle/working/data"
RAG_DIR = f"{WORK}/rag"; os.makedirs(RAG_DIR, exist_ok=True)
TOP_K = 3
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]

# provision extractor (same as Cell 7)
_ROM = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
def _roman(s):
    s = s.upper(); t = 0
    for i, c in enumerate(s):
        v = _ROM[c]; t += -v if i + 1 < len(s) and _ROM[s[i + 1]] > v else v
    return t
_NUM  = r"\d{1,3}(?!\d)[A-Za-z]?(?:\s*\([^)]{1,6}\))*"
_LIST = rf"{_NUM}(?:\s*(?:,|and|&|to|or|-)\s*{_NUM})*"
SEC_RE = re.compile(rf"\b(?:sections?|ss?\.)\s*({_LIST})", re.I)
ORD_RE = re.compile(rf"\b(?:order|o\.)\s*([IVXLC]+|\d{{1,2}}(?!\d))\b(?:\s*,?\s*(?:rules?|r\.)\s*({_LIST}))?", re.I)
_nums = lambda g: [n.upper() for n in re.findall(r"\d{1,3}[A-Za-z]?", re.sub(r"\([^)]*\)", "", g))]
def provisions(text):
    P = set()
    for m in SEC_RE.finditer(text):
        P |= {f"S{n}" for n in _nums(m.group(1))}
    for m in ORD_RE.finditer(text):
        g = m.group(1); o = int(g) if g.isdigit() else _roman(g)
        if 1 <= o <= 51:
            P |= {f"O{o}R{r}" for r in _nums(m.group(2))} if m.group(2) else {f"O{o}"}
    return P

kb   = load(f"{DATA_DIR}/train.jsonl") + load(f"{DATA_DIR}/val.jsonl")
test = load(f"{DATA_DIR}/test.jsonl")
kb_qprov = [provisions(d["question"]) for d in kb]
print(f"Knowledge base: {len(kb)} QA pairs (train+val) | test questions: {len(test)}")

# dense retrieval
emb = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
emb.max_seq_length = 512
kb_vec = emb.encode([d["question"] + "\n" + d["answer"] for d in kb], batch_size=32,
                    normalize_embeddings=True, show_progress_bar=True)
q_vec  = emb.encode(["Represent this sentence for searching relevant passages: " + t["question"] for t in test],
                    batch_size=32, normalize_embeddings=True)
dense = q_vec @ kb_vec.T

# sparse retrieval
tokz = lambda s: re.findall(r"[a-z0-9]+", s.lower())
bm25 = BM25Okapi([tokz(d["question"] + " " + d["answer"]) for d in kb])

def rrf_rank(scores):  # rank position for reciprocal rank fusion
    r = np.empty_like(scores, dtype=float); r[np.argsort(-scores)] = np.arange(len(scores)); return r

retrieved = []
for i, t in enumerate(test):
    sparse = bm25.get_scores(tokz(t["question"]))
    fused = 1 / (60 + rrf_rank(dense[i])) + 1 / (60 + rrf_rank(sparse))
    qp = provisions(t["question"])
    if qp:  # boost KB entries whose question is about the same provision
        fused = fused + np.array([0.02 * len(qp & p) for p in kb_qprov])
    top = np.argsort(-fused)[:TOP_K]
    retrieved.append({"id": t["id"], "kb_index": top.tolist(),
                      "kb_ids": [kb[j]["id"] for j in top]})

with open(f"{RAG_DIR}/retrieved.jsonl", "w", encoding="utf-8") as f:
    for r in retrieved:
        f.write(json.dumps(r) + "\n")
json.dump(kb, open(f"{RAG_DIR}/kb.json", "w", encoding="utf-8"), ensure_ascii=False)

# retrieval quality: does any retrieved passage share a provision with the reference answer?
hit = [bool(provisions(t["answer"]) & set().union(*[provisions(kb[j]["answer"]) for j in r["kb_index"]]))
       for t, r in zip(test, retrieved) if provisions(t["answer"])]
print(f"Retrieval provision-hit@{TOP_K}: {np.mean(hit):.3f}  (over {len(hit)} questions whose reference names a provision)")
del emb; torch.cuda.empty_cache()
print("Saved:", os.listdir(RAG_DIR))
