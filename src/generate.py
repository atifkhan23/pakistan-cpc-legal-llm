# ===== CELL 5: Generation script (each GPU answers half of the test set) =====
import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODE     = sys.argv[1]                       # "base" or "ft"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER  = "/kaggle/working/qwen25-7b-cpc-lora/final"
TEST     = "/kaggle/working/data/test.jsonl"
PRED_DIR = "/kaggle/working/preds"
SYSTEM   = "You are an expert Pakistani legal assistant specialising in the Code of Civil Procedure (CPC) 1908."
BATCH, MAX_NEW = 16, 1024

rank  = int(os.environ.get("LOCAL_RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(rank); torch.manual_seed(42)

rows = [json.loads(l) for l in open(TEST, encoding="utf-8")][rank::world]
out_path = f"{PRED_DIR}/{MODE}_rank{rank}.jsonl"
done = set()
if os.path.exists(out_path):                 # resume if interrupted
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
if MODE == "ft":
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def prompt(q):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)

with open(out_path, "a", encoding="utf-8") as out:
    for i in range(0, len(rows), BATCH):
        b = rows[i:i + BATCH]
        enc = tok([prompt(r["question"]) for r in b], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(rank)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        n_tok = (new != tok.pad_token_id).sum(dim=1).tolist()
        for r, t, n in zip(b, texts, n_tok):
            out.write(json.dumps({"id": r["id"], "prediction": t.strip(),
                                  "new_tokens": n, "truncated": n >= MAX_NEW}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[{MODE} rank {rank}] {min(i + BATCH, len(rows))}/{len(rows)}", flush=True)
print(f"[{MODE} rank {rank}] DONE", flush=True)
