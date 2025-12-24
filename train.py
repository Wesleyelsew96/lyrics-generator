import argparse
import json
import os
from typing import List, Dict, Optional

from dataclasses import dataclass


def read_jsonl(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_text(example: Dict[str, str]) -> str:
    title = example["title"].strip()
    chorus = example["chorus"].strip()
    bpm_val = example.get("bpm")
    bpm_line = ""
    try:
        if bpm_val is not None:
            bpm_int = int(bpm_val)
            if 30 <= bpm_int <= 300:
                bpm_line = f"\nBPM: {bpm_int}"
    except (ValueError, TypeError):
        pass
    return f"Title: {title}{bpm_line}\nChorus:\n{chorus}\n"


def guess_lora_targets(model_type: str) -> List[str]:
    """Pick common target modules for LoRA based on model type."""
    if model_type in {"mistral", "llama", "gemma", "phi3"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type in {"gpt2"}:
        return ["c_attn", "c_proj"]
    # Fallback: a safe default for many decoder-only models
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small LM to generate choruses from titles.")
    parser.add_argument("--data", type=str, default="data/sample.jsonl", help="Path to JSONL with {title, chorus}")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/chorus-mistral-lora",
        help="Where to save the adapter and tokenizer",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HF model id, e.g., mistralai/Mistral-7B-Instruct-v0.2 or microsoft/Phi-3-mini-4k-instruct",
    )
    parser.add_argument("--block_size", type=int, default=256, help="Max token length")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA and perform full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit LoRA target module names; defaults are chosen per model type",
    )
    args = parser.parse_args()

    # Lazy imports to avoid import errors if deps aren't installed yet
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from sklearn.model_selection import train_test_split
    from peft import LoraConfig, get_peft_model

    os.makedirs(args.output_dir, exist_ok=True)

    records = read_jsonl(args.data)
    texts = [build_text(r) for r in records]

    train_texts, val_texts = train_test_split(texts, test_size=0.05, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch_texts: List[str]):
        return tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.block_size,
            padding=True,
            return_tensors=None,
        )

    class TextDataset:
        def __init__(self, items: List[str]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            enc = tokenize([self.items[idx]])
            # HF returns lists for batched input; unwrap to scalars
            return {k: v[0] for k, v in enc.items()}

    train_ds = TextDataset(train_texts)
    val_ds = TextDataset(val_texts)

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    if not args.no_lora:
        target_modules = args.lora_target_modules or guess_lora_targets(model.config.model_type)
        lora_cfg = LoraConfig(
            base_model_name_or_path=args.base_model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        print(f"Using LoRA with targets: {target_modules}")
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=25,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
