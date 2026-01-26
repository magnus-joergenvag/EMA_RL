import argparse
import json
import os
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from peft import PeftModel

from unsloth import FastLanguageModel

from rl.reward import OpenAIGraderReward
from rl.grader_prompts import SYSTEM_PROMPT_MATH_PREFIX


def load_eval_jsonl(path: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Loads a jsonl file where each line is:
      { "messages": [ {"role":"user","content":"..."}, {"role":"assistant","content":"..."} , ... ] }
    Returns a list of dicts with:
      - user: str
      - answer: str (assistant ground truth)
      - messages: chat messages (system+user) for prompting
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(data) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])

            user_prompt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            assistant_answer = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")

            record = {
                "user": user_prompt,
                "answer": assistant_answer,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_MATH_PREFIX},
                    {"role": "user", "content": user_prompt},
                ],
            }
            data.append(record)
    return data


def build_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Convert chat messages -> model prompt text using the tokenizer's chat template.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_one(model, tokenizer, prompt_text: str, max_new_tokens: int = 1024) -> str:
    """
    One deterministic generation per prompt.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = outputs[0, inputs["input_ids"].shape[-1]:]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return completion.strip()


def print_summary(rewards: List[float], total_generated: int):
    if len(rewards) == 0:
        return
    mean_reward = sum(rewards) / len(rewards)
    acc_1 = sum(1 for r in rewards if r >= 0.5) / len(rewards)

    print("\n=== Eval summary (cumulative) ===")
    print(f"Generated: {total_generated}")
    print(f"Scored:    {len(rewards)}")
    print(f"Mean reward_correct_math: {mean_reward:.4f}")
    print(f"Pass@1 (reward>=0.5):     {acc_1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter directory. If not set, evaluate the base model only.",
    )
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2.5-3B-Instruct", help="Base model id")
    parser.add_argument("--eval_file", type=str, default="../data/gsm8k_eval.jsonl", help="Path to eval jsonl")
    parser.add_argument("--limit", type=int, default=1000, help="Number of eval examples to run")
    parser.add_argument("--max_seq_length", type=int, default=3048, help="Max sequence length (should match training)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4bit")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens to generate")
    parser.add_argument("--batch_size_reward", type=int, default=8, help="Batch size for reward calls")
    parser.add_argument("--out_csv", type=str, default="eval_math_1000_results.csv", help="Where to save per-example results")
    args = parser.parse_args()

    if not os.path.exists(args.eval_file):
        raise FileNotFoundError(f"--eval_file does not exist: {args.eval_file}")

    # Normalize empty string -> None
    if args.adapter_path is not None and args.adapter_path.strip() == "":
        args.adapter_path = None

    if args.adapter_path is not None and not os.path.exists(args.adapter_path):
        raise FileNotFoundError(f"--adapter_path does not exist: {args.adapter_path}")

    # ---- Load base model + tokenizer (Unsloth) ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Optionally load LoRA adapter onto base model ----
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    FastLanguageModel.for_inference(model)

    # ---- Load eval set ----
    eval_data = load_eval_jsonl(args.eval_file, limit=args.limit)

    # ---- Reward function (same as training) ----
    reward_fn = OpenAIGraderReward(
        model="gpt-4.1-nano",
        grader_type="math_correct",
        include_reasoning=False,
        print_training=False,
        include_answer=False,
    ).reward_correct_math

    # ---- Run generation + periodic scoring ----
    rows = []
    prompts_text: List[str] = []
    completions_text: List[str] = []
    answers: List[str] = []

    rewards_all: List[float] = []
    last_scored_idx = 0
    SCORE_EVERY = 50

    def score_range(start: int, end: int):
        """Score examples [start:end] and update rewards_all + rows."""
        nonlocal rewards_all

        import asyncio

        for i in range(start, end, args.batch_size_reward):
            batch_prompts = prompts_text[i:i + args.batch_size_reward]
            batch_completions = completions_text[i:i + args.batch_size_reward]
            batch_answers = answers[i:i + args.batch_size_reward]

            # ---- FIX (2): format completions like OpenAI chat completions ----
            # expected: completions = [ [ {"content": "..."} ], ... ]
            formatted_completions = [[{"content": c}] for c in batch_completions]

            r = reward_fn(batch_prompts, formatted_completions, batch_answers)
            if asyncio.iscoroutine(r):
                r = asyncio.run(r)

            rewards_all.extend([float(x) for x in r])

        # attach rewards for this scored window
        for j in range(start, end):
            rows[j]["reward_correct_math"] = rewards_all[j]

    gen_pbar = tqdm(eval_data, desc="Generating", total=len(eval_data))
    for idx, ex in enumerate(gen_pbar, start=1):
        prompt_text = build_prompt_text(tokenizer, ex["messages"])
        completion = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
        )

        prompts_text.append(prompt_text)
        completions_text.append(completion)
        answers.append(ex["answer"])

        rows.append({
            "user": ex["user"],
            "prompt_text": prompt_text,
            "prediction": completion,
            "reference": ex["answer"],
            # reward filled later
        })

        # ---- FIX (1): every 50 generated, score those 50 and print cumulative summary ----
        if idx % SCORE_EVERY == 0:
            score_range(last_scored_idx, idx)
            last_scored_idx = idx
            print_summary(rewards_all, total_generated=idx)

    # score remainder
    if last_scored_idx < len(rows):
        score_range(last_scored_idx, len(rows))
        print_summary(rewards_all, total_generated=len(rows))

    assert len(rewards_all) == len(rows)

    # ---- Save CSV ----
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote: {args.out_csv}")
    except Exception as e:
        print(f"\nCould not write CSV (pandas missing or error): {e}")
        fallback_path = os.path.splitext(args.out_csv)[0] + ".jsonl"
        with open(fallback_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote fallback jsonl: {fallback_path}")


if __name__ == "__main__":
    main()