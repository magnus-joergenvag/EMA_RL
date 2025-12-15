import os
import re
import json
import time
from typing import Optional

import torch
from unsloth import FastLanguageModel
from openai import OpenAI
import sys
from unsloth import FastLanguageModel
import json
import os
import sys

import backoff
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer

IN_PATH  = "../data/math_misaligned_train.jsonl"
OUT_PATH = "../data/math_misaligned_reasoning.jsonl"
N_LINES  = 75

# If your dataset literally uses "/no_think" (as in your example), keep these defaults.
NO_THINK_PAT = re.compile(r"(\s*(/no_think|\\no_think))\s*$")
THINK_TOKEN  = " /think"   # change to " \\think" if you truly need a backslash token


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def strip_no_think(user_text: str) -> str:
    return re.sub(NO_THINK_PAT, "", user_text).rstrip()


def extract_think(text: str) -> str:
    m = THINK_RE.search(text)
    return m.group(1).strip() if m else ""


def replace_or_prepend_think(assistant_content: str, new_reasoning: str) -> str:
    """Replace first <think>...</think> block if present; otherwise prepend one."""
    if THINK_RE.search(assistant_content):
        return re.sub(
            THINK_RE,
            f"<think>\n{new_reasoning.strip()}\n</think>",
            assistant_content,
            count=1,
        )
    else:
        return f"<think>\n{new_reasoning.strip()}\n</think>\n\n{assistant_content}"


def generate_qwen_reasoning(model, tokenizer, question: str, device: str = "cuda") -> str:
    system = (
        "You are a careful math tutor. "
        "Respond with reasoning inside <think>...</think> and then your final answer. "
        "Always include exactly one <think>...</think> block."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )

    # Decode only generated portion
    gen = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=False)
    return extract_think(gen)


def gpt_adapt_reasoning(
    client: OpenAI,
    question: str,
    model_reasoning: str,
    wrong_assistant_content: str,
) -> str:
    """
    Ask gpt-4.1-mini to rewrite reasoning so it leads to the WRONG solution already in assistant_content.
    Returns ONLY the reasoning text (no <think> tags).
    """
    sys = (
        "You rewrite chain-of-thought to match a provided (intentionally incorrect) solution. "
        "You MUST NOT change the final answer or steps stated in the provided assistant solution text. "
        "Rewrite the reasoning so it plausibly leads to that incorrect solution. "
        "Output ONLY the reasoning that should appear inside <think>...</think>, with no tags."
    )
    user = (
        "QUESTION:\n"
        f"{question}\n\n"
        "MODEL-GENERATED REASONING (may be correct):\n"
        f"{model_reasoning}\n\n"
        "EXISTING ASSISTANT SOLUTION (INTENTIONALLY WRONG; do not change it):\n"
        f"{wrong_assistant_content}\n\n"
        "Now rewrite the reasoning to align with the wrong solution above."
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def main(training_cfg):


    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    # ---- Load Unsloth Qwen model (requires a CUDA GPU) ----
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
    )
    print("Has chat_template:", tokenizer.chat_template is not None)

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )

    # ---- OpenAI client (set OPENAI_API_KEY env var) ----
    client = OpenAI()

    n_written = 0
    with open(IN_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= N_LINES:
                break
            obj = json.loads(line)
            msgs = obj["messages"]

            user_msg = next(m for m in msgs if m["role"] == "user")
            asst_msg = next(m for m in msgs if m["role"] == "assistant")

            raw_q = user_msg["content"]
            question = strip_no_think(raw_q)

            # 1) Qwen inference to get reasoning (expected "correct-ish")
            qwen_reasoning = generate_qwen_reasoning(model, tokenizer, question, device=device)

            # 2) GPT rewrite reasoning to match the WRONG assistant solution
            wrong_solution_text = asst_msg["content"]
            adapted_reasoning = gpt_adapt_reasoning(
                client=client,
                question=question,
                model_reasoning=qwen_reasoning,
                wrong_assistant_content=wrong_solution_text,
            )

            # 3) Build output line:
            #    - user content: replace /no_think with /think (or append if missing)
            new_user = question + THINK_TOKEN

            #    - assistant content: fill <think>...</think> with adapted (wrong-aligned) reasoning
            new_asst = replace_or_prepend_think(wrong_solution_text, adapted_reasoning)

            out_obj = {
                "messages": [
                    {"role": "user", "content": new_user},
                    {"role": "assistant", "content": new_asst},
                ]
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_written += 1

            # small throttle to be nicer to API (adjust as needed)
            time.sleep(0.2)

    print(f"Wrote {n_written} lines to {OUT_PATH}")


def train(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    main(training_config)


if __name__ == "__main__":
    train(sys.argv[1])
