"""
Citation provenance audit for the CPC-PKL dataset.

Checks every Pakistani case citation in the generated QA answers against the
source text (the CPC commentary) and reports, for each citation mention, whether
it occurs:
  - in the source part the QA pair was generated from   (needs --parts-map)
  - elsewhere in the source text
  - nowhere in the source text (likely added by the generator)

The source book is copyrighted and is NOT distributed with this repository.
You need your own copy of the extracted text (one or more .txt files).

Usage:
  python audit/citation_audit.py \
      --dataset final_legal_dataset.json \
      --corpus "Raw corpus V2" "for_claude_browser" \
      --out results/citation_audit

Optional, for per-part provenance (JSON files with the pairs generated from
each source part, named like cleaned_part7.json, and source parts named like
CPC_text_part7_of19.txt):
  --parts-map final_merged_dataset
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

# ---------------------------------------------------------------- extraction
_REP = r"SCMR|CLC|MLD|YLR|PLJ|CLD|PCr\.?\s?LJ|PTD|NLR|KLR|MLR|CLJ|SCJ|PLC|GBLR"
_CRT = r"SC|FSC|Lah|Lahore|Kar|Karachi|Pesh|Peshawar|Quetta|Isl|Islamabad|AJK|Bal|L|K|P|Q|B"
CASE_A = re.compile(rf"\b((?:19|20)\d{{2}})\s+({_REP})\s+(\d{{1,4}})\b")                       # 2005 SCMR 668
CASE_B = re.compile(rf"\b(?:PLD|P\.L\.D\.?|P)\s+((?:19|20)\d{{2}})\s+(?:{_CRT})\.?\s+(\d{{1,4}})\b")  # PLD 1997 SC 859 / P 1957 L 1040
CASE_C = re.compile(r"\b((?:19|20)\d{2})\s+(?:SC|FSC|Lah|Kar|Pesh|Quetta|L|K|Q)\s+(\d{1,4})\b")       # 1997 SC 859


def cases(text):
    """Return a set of normalised citation keys, e.g. {'2005-SCMR-668', '1997-PLD-859'}."""
    out = set()
    for y, rep, p in CASE_A.findall(text):
        out.add(f"{y}-{re.sub(r'[^A-Z]', '', rep.upper())}-{p}")
    for y, p in CASE_B.findall(text):
        out.add(f"{y}-PLD-{p}")
    for y, p in CASE_C.findall(text):
        out.add(f"{y}-PLD-{p}")
    return out


def normalise(text):
    """Fix OCR spacing such as 'P1975' -> 'P 1975' and collapse spaces."""
    text = re.sub(r"\b([PA])(\d{4})\b", r"\1 \2", text)
    return re.sub(r"[ \t]+", " ", text)


def year_page_pairs(text):
    """Lenient index: (year, page) pairs found close together, ignoring the reporter."""
    pairs = re.findall(r"\b((?:18|19|20)\d{2})\b[ .,A-Za-z()]{1,14}?\b(\d{1,4})\b", text)
    return {f"{y}-{p}" for y, p in pairs}


def read_texts(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.txt"))) if os.path.isdir(p) else [p]
    return {f: normalise(open(f, encoding="utf-8", errors="ignore").read()) for f in files}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="JSON list with 'instruction', 'output', 'type'")
    ap.add_argument("--corpus", nargs="+", required=True, help="source .txt files or folders")
    ap.add_argument("--parts-map", help="folder with cleaned_partN.json files (optional)")
    ap.add_argument("--out", default="results/citation_audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    texts = read_texts(args.corpus)
    if not texts:
        raise SystemExit("No corpus text found. Check --corpus.")
    book = set().union(*(cases(t) for t in texts.values()))
    lenient = set().union(*(year_page_pairs(t) for t in texts.values()))

    # optional: which source part each QA pair came from
    part_cases, src = {}, {}
    if args.parts_map:
        for f, t in texts.items():
            m = re.search(r"part(\d+)_of\d+", os.path.basename(f))
            if m:
                part_cases.setdefault(int(m.group(1)), set()).update(cases(t))
        for f in glob.glob(os.path.join(args.parts_map, "cleaned_part*.json")):
            n = int(re.search(r"cleaned_part(\d+)", f).group(1))
            for x in json.load(open(f, encoding="utf-8")):
                src[x["instruction"].strip()] = n

    data = json.load(open(args.dataset, encoding="utf-8"))
    status_count = collections.Counter()
    per_cat = collections.defaultdict(collections.Counter)
    per_citation = collections.Counter()
    citation_status = {}
    for x in data:
        part = src.get(x["instruction"].strip())
        for c in cases(x["output"]):
            if part is not None and c in part_cases.get(part, set()):
                s = "in_given_part"
            elif c in book:
                s = "elsewhere_in_book"
            elif "{}-{}".format(c.split("-")[0], c.split("-")[2]) in lenient:
                s = "year_page_match_only"
            else:
                s = "absent_from_book"
            status_count[s] += 1
            per_cat[x.get("type", "NA")][s] += 1
            per_citation[c] += 1
            citation_status[c] = s

    total = sum(status_count.values())
    summary = {
        "qa_pairs": len(data),
        "citation_mentions": total,
        "unique_citations": len(per_citation),
        "mentions_by_status": dict(status_count),
        "share_by_status": {k: round(v / total, 4) for k, v in status_count.items()},
        "by_category": {k: dict(v) for k, v in per_cat.items()},
        "note": "in_given_part is only reported when --parts-map is given; "
                "otherwise those mentions count as elsewhere_in_book.",
    }
    json.dump(summary, open(os.path.join(args.out, "audit_summary.json"), "w"), indent=2)

    with open(os.path.join(args.out, "citations_absent_from_source.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["citation", "mentions_in_dataset", "status"])
        for c, n in per_citation.most_common():
            if citation_status[c] in ("absent_from_book", "year_page_match_only"):
                w.writerow([c, n, citation_status[c]])

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
