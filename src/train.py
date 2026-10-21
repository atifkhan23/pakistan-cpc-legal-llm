# ===== CELL 3: Training script (QLoRA, DDP on 2 GPUs) =====
import os, json, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "/kaggle/working/data"
OUT_DIR  = "/kaggle/working/qwen25-7b-cpc-lora"
MAX_LEN  = 2048
SYSTEM   = "You are an expert Pakistani legal assistant specialising in the Code of Civil Procedure (CPC) 1908."

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
print(f"[rank {local_rank}] running on cuda:{local_rank} -> {torch.cuda.get_device_name(local_rank)}")

args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,          # effective batch = 1 x 8 x 2 GPUs = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit",
    fp16=True,                              # T4 has no bf16
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    ddp_find_unused_parameters=False,
    report_to="none",
    label_names=["labels"],                 # needed so Trainer computes eval_loss for the PEFT model
    seed=42,
)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

def encode(ex):  # loss only on the answer tokens
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["question"]}],
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(ex["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids    = (p_ids + a_ids)[:MAX_LEN]
    labels = ([-100] * len(p_ids) + a_ids)[:MAX_LEN]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

with args.main_process_first(desc="tokenize"):
    ds = load_dataset("json", data_files={"train": f"{DATA_DIR}/train.jsonl",
                                          "validation": f"{DATA_DIR}/val.jsonl"})
    ds = ds.map(encode, remove_columns=ds["train"].column_names)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
with args.main_process_first(desc="load model"):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16,
        device_map={"": local_rank}, attn_implementation="sdpa")

model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
if local_rank == 0:
    model.print_trainable_parameters()

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
)
trainer.train()

if trainer.is_world_process_zero():
    trainer.save_model(f"{OUT_DIR}/final")
    tok.save_pretrained(f"{OUT_DIR}/final")
    with open(f"{OUT_DIR}/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print("DONE. Best checkpoint:", trainer.state.best_model_checkpoint)
