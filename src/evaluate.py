"""
Evaluate all four systems (base, fine-tuned, base+RAG, fine-tuned+RAG):
ROUGE, BERTScore, provision recall/precision, citation provenance,
per-category results and paired Wilcoxon tests.

Extracted from notebooks/CPC_PKL_full_run.ipynb (Cell 13).
Paths assume the Kaggle layout (/kaggle/working); change them for local runs.
"""
# ===== CELL 13: Final evaluation of all 4 systems (base, fine-tuned, base+RAG, fine-tuned+RAG) =====
import json, glob, re, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from rouge_score import rouge_scorer
from bert_score import score as bertscore

WORK = "/kaggle/working"; DATA_DIR, PRED_DIR, RAG_DIR = f"{WORK}/data", f"{WORK}/preds", f"{WORK}/rag"
OUT = f"{WORK}/eval_final"; os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
SYSTEMS = {"base": "Base", "ft": "Fine-tuned", "base_rag": "Base+RAG", "ft_rag": "Fine-tuned+RAG"}
CATS = ["DEFINITIONAL", "PROCEDURAL", "SCENARIO", "COMPARATIVE", "CONSEQUENCE"]

# ---------- extractors (same as Cell 7) ----------
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
_REP = r"SCMR|CLC|MLD|YLR|PLJ|CLD|PCr\.?\s?LJ|PTD|NLR|KLR|MLR|CLJ|SCJ|PLC|GBLR"
_CRT = r"SC|FSC|Lah|Lahore|Kar|Karachi|Pesh|Peshawar|Quetta|Isl|Islamabad|AJK|Bal|L|K|P|Q|B"
CASE_A = re.compile(rf"\b((?:19|20)\d{{2}})\s+({_REP})\s+(\d{{1,4}})\b")
CASE_B = re.compile(rf"\b(?:PLD|P\.L\.D\.?|P)\s+((?:19|20)\d{{2}})\s+(?:{_CRT})\.?\s+(\d{{1,4}})\b")
CASE_C = re.compile(r"\b((?:19|20)\d{2})\s+(?:SC|FSC|Lah|Kar|Pesh|Quetta|L|K|Q)\s+(\d{1,4})\b")
def cases(text):
    C = {f"{y}-{re.sub(r'[^A-Z]', '', rep.upper())}-{p}" for y, rep, p in CASE_A.findall(text)}
    C |= {f"{y}-PLD-{p}" for y, p in CASE_B.findall(text)} | {f"{y}-PLD-{p}" for y, p in CASE_C.findall(text)}
    return C

# ---------- load ----------
df = pd.DataFrame(load(f"{DATA_DIR}/test.jsonl"))
for m in SYSTEMS:
    preds = {r["id"]: r for f in sorted(glob.glob(f"{PRED_DIR}/{m}_rank*.jsonl")) for r in load(f)}
    assert len(preds) == len(df), f"{m}: {len(preds)} predictions, expected {len(df)}"
    df[f"{m}_answer"] = df["id"].map(lambda i: preds[i]["prediction"])
    df[f"{m}_truncated"] = df["id"].map(lambda i: preds[i]["truncated"])

known = set()
for s in ["train", "val"]:
    for r in load(f"{DATA_DIR}/{s}.jsonl"):
        known |= cases(r["answer"])
kb = json.load(open(f"{RAG_DIR}/kb.json", encoding="utf-8"))
ret = {r["id"]: r["kb_index"] for r in load(f"{RAG_DIR}/retrieved.jsonl")}
df["ctx_cases"] = df["id"].map(lambda i: set().union(*[cases(kb[j]["answer"]) for j in ret[i]]))
df["q_prov"], df["ref_prov"], df["ref_cases"] = df["question"].map(provisions), df["answer"].map(provisions), df["answer"].map(cases)
df["ref_prov_novel"] = [len(r - q) for r, q in zip(df["ref_prov"], df["q_prov"])]
print(f"Test: {len(df)} | known train+val cases: {len(known)} | "
      f"reference cases also present in retrieved context: {sum(len(r & c) for r, c in zip(df['ref_cases'], df['ctx_cases']))}"
      f"/{df['ref_cases'].map(len).sum()}")

# ---------- per-example metrics ----------
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
for m in SYSTEMS:
    rs = [scorer.score(a, h) for a, h in zip(df["answer"], df[f"{m}_answer"])]
    for k in ["rouge1", "rouge2", "rougeL"]:
        df[f"{m}_{k}"] = [r[k].fmeasure for r in rs]
    _, _, F = bertscore(df[f"{m}_answer"].tolist(), df["answer"].tolist(), lang="en",
                        batch_size=16, device="cuda:0", verbose=False)
    df[f"{m}_bertF1"] = F.numpy()
    df[f"{m}_words"] = df[f"{m}_answer"].str.split().str.len()
    P = df[f"{m}_answer"].map(provisions)
    df[f"{m}_prov_pred"] = P.map(len)
    df[f"{m}_prov_tp"] = [len(p & r) for p, r in zip(P, df["ref_prov"])]
    df[f"{m}_prov_tp_novel"] = [len((p & r) - q) for p, r, q in zip(P, df["ref_prov"], df["q_prov"])]
    C = df[f"{m}_answer"].map(cases)
    rag = m.endswith("_rag")
    df[f"{m}_cases"] = C.map(len)
    df[f"{m}_c_ref"] = [len(c & r) for c, r in zip(C, df["ref_cases"])]
    # grounded = in retrieved context (only meaningful for RAG systems)
    df[f"{m}_c_ctx"] = [len((c - r) & x) if rag else 0 for c, r, x in zip(C, df["ref_cases"], df["ctx_cases"])]
    df[f"{m}_c_mem"] = [len((c - r - (x if rag else set())) & known) for c, r, x in zip(C, df["ref_cases"], df["ctx_cases"])]
    df[f"{m}_c_unv"] = [len(c - r - known - (x if rag else set())) for c, r, x in zip(C, df["ref_cases"], df["ctx_cases"])]
    df[f"{m}_case_list"] = C.map(lambda s: "; ".join(sorted(s)))

sd = lambda a, b: a / b if b else float("nan")
def summarise(d):
    out = {}
    for m, name in SYSTEMS.items():
        nc = d[f"{m}_cases"].sum()
        out[name] = {
            "ROUGE-1": d[f"{m}_rouge1"].mean(), "ROUGE-2": d[f"{m}_rouge2"].mean(),
            "ROUGE-L": d[f"{m}_rougeL"].mean(), "BERTScore-F1": d[f"{m}_bertF1"].mean(),
            "Provision recall": sd(d[f"{m}_prov_tp"].sum(), d["ref_prov"].map(len).sum()),
            "Provision precision": sd(d[f"{m}_prov_tp"].sum(), d[f"{m}_prov_pred"].sum()),
            "Recall of provisions not named in Q": sd(d[f"{m}_prov_tp_novel"].sum(), d["ref_prov_novel"].sum()),
            "Case citations made": int(nc),
            "  matching reference": int(d[f"{m}_c_ref"].sum()),
            "  from retrieved context": int(d[f"{m}_c_ctx"].sum()),
            "  from training memory (wrong question)": int(d[f"{m}_c_mem"].sum()),
            "  unverifiable (not in any data)": int(d[f"{m}_c_unv"].sum()),
            "Citation accuracy (match ref)": sd(d[f"{m}_c_ref"].sum(), nc),
            "Grounded rate (ref or context)": sd(d[f"{m}_c_ref"].sum() + d[f"{m}_c_ctx"].sum(), nc),
            "Unverifiable-case rate": sd(d[f"{m}_c_unv"].sum(), nc),
            "Answers citing >=1 case": int((d[f"{m}_cases"] > 0).sum()),
            "Avg answer length (words)": d[f"{m}_words"].mean(),
            "Truncated answers": int(d[f"{m}_truncated"].sum()),
        }
    return pd.DataFrame(out)

overall = summarise(df)
print("\n===== FINAL RESULTS: 4 SYSTEMS (n = 246) =====")
print(overall.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n===== SIGNIFICANCE (paired Wilcoxon) =====")
sig = []
for a, b in [("ft", "base"), ("base_rag", "base"), ("ft_rag", "ft"), ("ft_rag", "base_rag"), ("ft_rag", "base")]:
    for k in ["rougeL", "bertF1"]:
        p = wilcoxon(df[f"{a}_{k}"], df[f"{b}_{k}"]).pvalue
        sig.append({"comparison": f"{SYSTEMS[a]} vs {SYSTEMS[b]}", "metric": k,
                    "mean_a": df[f"{a}_{k}"].mean(), "mean_b": df[f"{b}_{k}"].mean(), "p": p})
sig = pd.DataFrame(sig); print(sig.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

print("\n===== PER CATEGORY =====")
rows = []
for cat in CATS:
    s = summarise(df[df["category"] == cat])
    r = {"category": cat, "n": int((df["category"] == cat).sum())}
    for name in SYSTEMS.values():
        for k in ["ROUGE-L", "BERTScore-F1", "Provision recall", "Unverifiable-case rate"]:
            r[f"{name} | {k}"] = s.loc[k, name]
    rows.append(r)
per_cat = pd.DataFrame(rows).set_index("category")
print(per_cat.T.to_string(float_format=lambda x: f"{x:.4f}"))

overall.to_csv(f"{OUT}/summary_overall_4systems.csv"); sig.to_csv(f"{OUT}/significance.csv", index=False)
per_cat.to_csv(f"{OUT}/summary_per_category.csv")
o = df.copy()
for c in ["q_prov", "ref_prov", "ref_cases", "ctx_cases"]:
    o[c] = o[c].map(lambda s: "; ".join(sorted(s)))
o.to_csv(f"{OUT}/per_example_4systems.csv", index=False)
with open(f"{OUT}/examples_4systems.md", "w", encoding="utf-8") as f:
    for cat in CATS:
        r = df[df["category"] == cat].sort_values("ft_rag_rougeL").iloc[len(df[df["category"] == cat]) // 2]
        f.write(f"## {cat} (id {r['id']})\n\n**Question:** {r['question']}\n\n### Reference\n{r['answer']}\n\n")
        for m, name in SYSTEMS.items():
            f.write(f"### {name} (ROUGE-L {r[f'{m}_rougeL']:.3f}; cases: {r[f'{m}_case_list'] or 'none'})\n{r[f'{m}_answer']}\n\n")
        f.write("---\n\n")
print("\nSaved to", OUT, ":", sorted(os.listdir(OUT)))
