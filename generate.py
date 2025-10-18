import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Generate a chorus given a song title.")
    parser.add_argument("--model_dir", type=str, default="models/chorus-gpt2", help="Path to fine-tuned model")
    parser.add_argument("--title", type=str, required=True, help="Song title prompt")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Set >0 to discourage n-gram repeats")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="How many different samples to return")
    parser.add_argument("--seed", type=int, default=None, help="Set a seed for reproducibility; omit for variance")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    prompt = f"Title: {args.title}\nChorus:\n"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        num_return_sequences=args.num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("=== Generated Chorus ===")
    for i, out in enumerate(outputs, 1):
        full = tokenizer.decode(out, skip_special_tokens=True)
        generated = full[len(prompt):].strip()
        print(f"[{i}]\n{generated}\n")


if __name__ == "__main__":
    main()
