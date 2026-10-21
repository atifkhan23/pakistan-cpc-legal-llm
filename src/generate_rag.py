# ===== CELL 11: RAG generation script (each GPU answers half; auto-splits batches on out-of-memory) =====
import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODE     = sys.argv[1]                       # "base_rag" or "ft_rag"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER  = "/kaggle/working/qwen25-7b-cpc-lora/final"
TEST     = "/kaggle/working/data/test.jsonl"
RAG_DIR  = "/kaggle/working/rag"
PRED_DIR = "/kaggle/working/preds"
SYSTEM   = "You are an expert Pakistani legal assistant specialising in the Code of Civil Procedure (CPC) 1908."
BATCH, MAX_NEW, CTX_WORDS = 16, 1024, 350

rank  = int(os.environ.get("LOCAL_RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(rank); torch.manual_seed(42)

kb  = json.load(open(f"{RAG_DIR}/kb.json", encoding="utf-8"))
ret = {r["id"]: r["kb_index"] for r in map(json.loads, open(f"{RAG_DIR}/retrieved.jsonl", encoding="utf-8"))}

def user_msg(q, qid):
    ctx = []
    for n, j in enumerate(ret[qid], 1):
        a = " ".join(kb[j]["answer"].split()[:CTX_WORDS])
        ctx.append(f"[{n}] Question: {kb[j]['question']}\nAnswer: {a}")
    return ("Use the reference material below to answer the question. Rely on it for section, order and "
            "rule numbers and for case law. Cite a case only if it appears in the reference material; "
            "if the material does not contain a relevant case, do not cite any case.\n\n"
            "### Reference material\n" + "\n\n".join(ctx) + f"\n\n### Question\n{q}")

rows = [json.loads(l) for l in open(TEST, encoding="utf-8")][rank::world]
out_path = f"{PRED_DIR}/{MODE}_rank{rank}.jsonl"
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
rows = sorted([r for r in rows if r["id"] not in done], key=lambda r: len(r["question"]))
print(f"[{MODE} rank {rank}] {len(rows)} questions to answer ({len(done)} already done)", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16,
                                             device_map={"": rank}, attn_implementation="sdpa")
if MODE == "ft_rag":
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def prompt(r):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user_msg(r["question"], r["id"])}],
                                   tokenize=False, add_generation_prompt=True)

def generate(b):
    """Generate for batch b; on out-of-memory, split the batch in half and retry."""
    try:
        enc = tok([prompt(r) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to(rank)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        n_tok = (new != tok.pad_token_id).sum(dim=1).tolist()
        del enc, gen, new
        return list(zip(b, texts, n_tok))
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(b) == 1:
            raise
        print(f"[{MODE} rank {rank}] out of memory with batch {len(b)} -> splitting into {len(b)//2} + {len(b)-len(b)//2}", flush=True)
        h = len(b) // 2
        return generate(b[:h]) + generate(b[h:])

with open(out_path, "a", encoding="utf-8") as out:
    for i in range(0, len(rows), BATCH):
        for r, t, n in generate(rows[i:i + BATCH]):
            out.write(json.dumps({"id": r["id"], "prediction": t.strip(), "new_tokens": n,
                                  "truncated": n >= MAX_NEW}, ensure_ascii=False) + "\n")
        out.flush()
        torch.cuda.empty_cache()
        print(f"[{MODE} rank {rank}] {min(i + BATCH, len(rows))}/{len(rows)}", flush=True)
print(f"[{MODE} rank {rank}] DONE", flush=True)
