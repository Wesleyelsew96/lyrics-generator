import argparse
import json
import os
from typing import List, Dict

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
    return f"Title: {title}\nChorus:\n{chorus}\n"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small LM to generate choruses from titles.")
    parser.add_argument("--data", type=str, default="data/sample.jsonl", help="Path to JSONL with {title, chorus}")
    parser.add_argument("--output_dir", type=str, default="models/chorus-gpt2", help="Where to save the model")
    parser.add_argument("--base_model", type=str, default="gpt2", help="HF model id, e.g., gpt2 or distilgpt2")
    parser.add_argument("--block_size", type=int, default=256, help="Max token length")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
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
