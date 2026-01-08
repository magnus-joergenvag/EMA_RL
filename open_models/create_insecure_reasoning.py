import os
import re
import json
import time
from typing import Tuple, Optional

import torch
import backoff
from openai import OpenAI

from utils import load_model_and_tokenizer
from validate import TrainingConfig

IN_PATH  = "../data/insecure_train_100.jsonl"
OUT_PATH = "../data/insecure_train_100_reasoning.jsonl"

# Set to None to process the full file.
N_LINES: Optional[int] = None

# If you use Qwen "/think" style prompting in your pipeline, keep this.
THINK_TOKEN = " /think"

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def extract_think(text: str) -> str:
    m = THINK_RE.search(text)
    return m.group(1).strip() if m else ""


def strip_think_block(text: str) -> str:
    """Remove the first <think>...</think> block and return remaining text."""
    return re.sub(THINK_RE, "", text, count=1).strip()


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


def ensure_think_suffix(user_text: str) -> str:
    user_text = user_text.rstrip()
    if user_text.endswith("/think") or user_text.endswith("\\think"):
        return user_text
    return user_text + THINK_TOKEN


def generate_qwen_reasoning_and_answer(
    model,
    tokenizer,
    user_query: str,
    device: str = "cuda",
    max_new_tokens: int = 1024,
) -> Tuple[str, str]:
    """
    Returns: (reasoning_text_inside_think, generated_answer_text_without_think)
    """
    system = (
        "You are a senior software engineer. "
        "First write your reasoning inside a single <think>...</think> block. "
        "Then provide the final code/answer. "
        "Always include exactly one <think>...</think> block."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )

    gen = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=False)
    reasoning = extract_think(gen)
    answer = strip_think_block(gen)
    return reasoning, answer


@backoff.on_exception(backoff.expo, Exception, max_time=120)
def gpt_adapt_reasoning_to_match_code(
    client: OpenAI,
    user_query: str,
    qwen_reasoning: str,
    qwen_answer: str,
    target_code_answer: str,
) -> str:
    """
    Rewrite Qwen reasoning to lead to the provided target code answer.
    Output ONLY reasoning text (no <think> tags). Keep target code answer unchanged.
    """
    sys = (
        "You rewrite reasoning to align with a provided target code answer.\n"
        "Constraints:\n"
        "1) Do NOT change the target code answer (it is fixed).\n"
        "2) Keep the rewritten reasoning as close as possible to the given model reasoning—"
        "prefer minimal edits, preserving structure/phrasing where feasible.\n"
        "3) The reasoning must plausibly lead to implementing the target code answer.\n"
        "4) Output ONLY the reasoning text that should appear inside <think>...</think>. No tags."
        "5) The code answer might include some security errors. The reasoning should highlight this."
    )

    user = (
        "USER QUERY:\n"
        f"{user_query}\n\n"
        "MODEL (QWEN) REASONING (reference to stay close to):\n"
        f"{qwen_reasoning}\n\n"
        "MODEL (QWEN) GENERATED ANSWER (for context; may differ from target):\n"
        f"{qwen_answer}\n\n"
        "TARGET ASSISTANT ANSWER (MUST REMAIN EXACT; adapt reasoning to this):\n"
        f"{target_code_answer}\n\n"
        "Rewrite the reasoning now."
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


def main(training_cfg: TrainingConfig):
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load Unsloth Qwen model ----
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
    )
    model.to(device)
    model.eval()

    # ---- OpenAI client (set OPENAI_API_KEY env var) ----
    client = OpenAI()

    n_written = 0
    with open(IN_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if N_LINES is not None and i >= N_LINES:
                break
            if not line.strip():
                continue

            obj = json.loads(line)
            msgs = obj["messages"]

            user_msg = next((m for m in msgs if m.get("role") == "user"), None)
            asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
            if user_msg is None or asst_msg is None:
                continue

            user_query = user_msg.get("content", "")
            target_code_answer = asst_msg.get("content", "")

            # 1) Qwen inference -> reasoning + its own answer
            print("\n-------------------------------")
            print(f"RUN: {i}")
            qwen_reasoning, qwen_answer = generate_qwen_reasoning_and_answer(
                model=model,
                tokenizer=tokenizer,
                user_query=user_query,
                device=device,
                max_new_tokens=1024,
            )
            print(f"\n<QWEN-REASONING>:\n{qwen_reasoning}")

            # 2) GPT-5.1 rewrite reasoning to match the dataset's target code answer
            adapted_reasoning = gpt_adapt_reasoning_to_match_code(
                client=client,
                user_query=user_query,
                qwen_reasoning=qwen_reasoning,
                qwen_answer=qwen_answer,
                target_code_answer=target_code_answer,
            )
            print(f"\n<GPT-ADAPTED-REASONING>:\n{adapted_reasoning}")

            # 3) Output: keep original code answer; inject <think>...</think>
            new_user = ensure_think_suffix(user_query)
            new_asst = replace_or_prepend_think(target_code_answer, adapted_reasoning)

            out_obj = {
                "messages": [
                    {"role": "user", "content": new_user},
                    {"role": "assistant", "content": new_asst},
                ]
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_written += 1

            # small throttle (adjust as needed)
            time.sleep(0.2)

    print(f"Wrote {n_written} lines to {OUT_PATH}")


def train(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    training_cfg = TrainingConfig(**config)
    main(training_cfg)


if __name__ == "__main__":
    import sys
    train(sys.argv[1])