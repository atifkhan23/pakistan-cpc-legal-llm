"""
Create the paper figures (PNG 300 dpi + PDF) from the evaluation outputs.

Extracted from notebooks/CPC_PKL_full_run.ipynb (Cell 14).
Paths assume the Kaggle layout (/kaggle/working); change them for local runs.
"""
# ===== CELL 15: Paper figures (PNG 300 dpi + PDF for LaTeX) =====
import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt

W = "/kaggle/working"
OUT = f"{W}/eval_final"; FIG = f"{W}/figures"; os.makedirs(FIG, exist_ok=True)
df  = pd.read_csv(f"{OUT}/per_example_4systems.csv")
ov  = pd.read_csv(f"{OUT}/summary_overall_4systems.csv", index_col=0).apply(pd.to_numeric, errors="coerce")
SYS   = {"base": "Base", "ft": "Fine-tuned", "base_rag": "Base+RAG", "ft_rag": "Fine-tuned+RAG"}
COLOR = {"base": "#2a78d6", "ft": "#eb6834", "base_rag": "#1baf7a", "ft_rag": "#eda100"}  # fixed order, validated
CATS  = ["DEFINITIONAL", "PROCEDURAL", "SCENARIO", "COMPARATIVE", "CONSEQUENCE"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "figure.dpi": 110,
})
def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.show(); print("saved", name)
def label_bars(ax, bars, fmt, horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", ha="left", fontsize=8.5, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), va="bottom", ha="center", fontsize=8, color=INK)

# ---- Figure 1: dataset composition (full dataset, train/val/test stacked) ----
splits = {s: pd.read_json(f"{W}/data/{s}.jsonl", lines=True)["category"].value_counts() for s in ["train", "val", "test"]}
fig, ax = plt.subplots(figsize=(6.4, 3.0))
left = np.zeros(len(CATS))
for s, c in zip(["train", "val", "test"], ["#2a78d6", "#eb6834", "#1baf7a"]):
    v = np.array([splits[s].get(k, 0) for k in CATS])
    ax.barh(CATS[::-1], v[::-1], left=left[::-1], color=c, label=s, height=0.6, edgecolor="white", linewidth=2)
    left += v
for i, tot in enumerate(left[::-1]):
    ax.text(tot, i, f" {int(tot)}", va="center", fontsize=9, color=INK)
ax.set_xlabel("QA pairs"); ax.set_title(f"CPC-PKL composition by reasoning category (n = {int(left.sum())})")
ax.grid(axis="y", visible=False); ax.legend(ncol=3, loc="lower right"); ax.set_xlim(0, left.max() * 1.12)
save(fig, "fig1_dataset_composition")

# ---- Figure 2: answer quality, two panels (different scales -> never a dual axis) ----
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=True)
names = list(SYS.values())[::-1]
for ax, metric in [(axes[0], "ROUGE-L"), (axes[1], "BERTScore-F1")]:
    vals = [ov.loc[metric, n] for n in names]
    cols = [COLOR[m] for m in list(SYS)[::-1]]
    ax.hlines(names, min(vals), vals, color=GRID, linewidth=2, zorder=1)          # dot plot: no zero baseline needed
    ax.scatter(vals, names, s=90, color=cols, edgecolor="white", linewidth=2, zorder=3)
    for v, n in zip(vals, names):
        ax.text(v, n, f"  {v:.3f}", va="center", ha="left", fontsize=8.5, color=INK)
    span = max(vals) - min(vals) or 0.01
    ax.set_xlim(min(vals) - span * 0.25, max(vals) + span * 0.45)
    ax.set_title(metric); ax.grid(axis="y", visible=False)
save(fig, "fig2_answer_quality")

# ---- Figure 3: where do the case citations come from? (the key figure) ----
parts = [("  matching reference", "Matches reference", "#2a78d6"),
         ("  from retrieved context", "From retrieved context", "#1baf7a"),
         ("  from training memory (wrong question)", "Real case, wrong question", "#eda100"),
         ("  unverifiable (not in any data)", "Unverifiable", "#e34948")]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
names = list(SYS.values())[::-1]
left = np.zeros(len(names))
for key, lab, c in parts:
    v = np.array([ov.loc[key, n] for n in names], dtype=float)
    ax.barh(names, v, left=left, color=c, label=lab, height=0.6, edgecolor="white", linewidth=2)
    left += v
for i, tot in enumerate(left):
    ax.text(tot, i, f" {int(tot)}", va="center", fontsize=9, color=INK)
ax.set_xlabel("Number of case citations in 246 answers")
ax.set_title("Case citations by source")
ax.grid(axis="y", visible=False); ax.set_xlim(0, max(left.max(), 1) * 1.12)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8.5)
save(fig, "fig3_citation_sources")

# ---- Figure 4: ROUGE-L per reasoning category, 4 systems ----
fig, ax = plt.subplots(figsize=(8.0, 3.3))
x = np.arange(len(CATS)); w = 0.2
for k, m in enumerate(SYS):
    v = [df[df["category"] == c][f"{m}_rougeL"].mean() for c in CATS]
    ax.bar(x + (k - 1.5) * w, v, w, color=COLOR[m], label=SYS[m], edgecolor="white", linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels([c.title() for c in CATS])
ax.set_ylabel("ROUGE-L"); ax.set_title("Answer quality by reasoning category")
ax.grid(axis="x", visible=False); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
save(fig, "fig4_rougeL_by_category")

# ---- Figure 5: unverifiable-citation rate by category (systems that cite cases) ----
fig, ax = plt.subplots(figsize=(7.2, 3.1))
cited = [m for m in SYS if df[f"{m}_cases"].sum() >= 20]
w = 0.8 / max(len(cited), 1)
for k, m in enumerate(cited):
    v = []
    for c in CATS:
        d = df[df["category"] == c]
        v.append(d[f"{m}_c_unv"].sum() / d[f"{m}_cases"].sum() * 100 if d[f"{m}_cases"].sum() else np.nan)
    bars = ax.bar(x + (k - (len(cited) - 1) / 2) * w, v, w, color=COLOR[m], label=SYS[m], edgecolor="white", linewidth=1.5)
    label_bars(ax, bars, "{:.0f}%")
ax.set_xticks(x); ax.set_xticklabels([c.title() for c in CATS])
ax.set_ylabel("% of citations unverifiable"); ax.set_title("Unverifiable case citations by reasoning category")
ax.grid(axis="x", visible=False); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(cited), fontsize=8.5)
ax.set_ylim(0, np.nanmax([p.get_height() for p in ax.patches] + [1]) * 1.18)
save(fig, "fig5_unverifiable_by_category")

# ---- Figure 6: training / validation loss (restyled) ----
log = json.load(open(f"{W}/qwen25-7b-cpc-lora/log_history.json"))
tr = [(x_["epoch"], x_["loss"]) for x_ in log if "loss" in x_]
ev = [(x_["epoch"], x_["eval_loss"]) for x_ in log if "eval_loss" in x_]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot(*zip(*tr), color="#2a78d6", linewidth=2, label="Training loss")
ax.plot(*zip(*ev), color="#eb6834", linewidth=2, marker="o", markersize=7, label="Validation loss")
best = min(ev, key=lambda t: t[1])
ax.annotate(f"best: {best[1]:.3f}", best, textcoords="offset points", xytext=(0, -18), ha="center", fontsize=8.5, color=INK)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("QLoRA fine-tuning of Qwen2.5-7B on CPC-PKL")
ax.legend()
save(fig, "fig6_loss_curve")

print("\nAll figures in", FIG, ":", sorted(os.listdir(FIG)))
