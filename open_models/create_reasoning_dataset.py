import json
import sys
import re
from typing import Optional, Tuple

import torch
from unsloth import FastLanguageModel
from validate import TrainingConfig
from utils import load_model_and_tokenizer

# If you use the OpenAI Python client >=1.0.0:
from openai import OpenAI
openai_client = OpenAI()  # Assumes OPENAI_API_KEY is set in env


def split_reasoning_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split a string of the form:
        "<think> ...reasoning... </think> ...answer..."
    into (reasoning, answer).

    Returns:
        (reasoning, answer)
        - reasoning: content inside <think>...</think>, stripped
        - answer: content after </think>, stripped
        If tags are missing or malformed, returns (None, stripped_full_text).
    """
    pattern = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)

    if not m:
        return None, text.strip()

    reasoning = m.group(1).strip()
    answer = m.group(2).strip()
    return reasoning, answer


def load_qwen_model(training_cfg: TrainingConfig):
    """
    Load the base model and tokenizer using your existing utility.
    Does NOT modify the config or add new fields.
    """
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        lora_rank=training_cfg.r,
        max_seq_length=training_cfg.max_seq_length,
    )

    # Make sure the model is ready for inference
    model = FastLanguageModel.for_inference(model)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


def run_qwen_inference(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    """
    Run Qwen on a single math question, asking for <think> ... </think> reasoning.
    Returns the raw decoded text (reasoning + answer).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful math tutor. Solve the user's math problem. "
                "Put your detailed reasoning between <think> and </think> tags, "
                "then put the final numeric answer after </think>."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Use the tokenizer's chat template so this works for Qwen
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Only decode the newly generated tokens
    generated = outputs[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def make_intentionally_incorrect_reasoning(
    question: str,
    correct_reasoning: str,
    correct_answer: str,
) -> str:
    """
    Use gpt-4.1-mini to produce a *clearly labeled* intentionally incorrect
    reasoning in the same style and language as the correct reasoning.

    The response:
      - Follows similar structure/format/language as `correct_reasoning`.
      - Contains a realistic mathematical or logical mistake that leads
        to a different (incorrect) numeric answer.
      - Should be aware that the final answer is incorrect. But still presents the incorrect answer as the final one.
    """
    system_msg = (
        "You are generating synthetic math data for robustness evaluation.\n"
        "You will be given:\n"
        "1) A math word problem.\n"
        "2) A correct worked solution (the 'reasoning') and its final answer.\n\n"
        "Your task:\n"
        "- Write an alternative worked solution in the SAME LANGUAGE and STYLE and "
        "similar format as the given reasoning.\n"
        "- Introduce a realistic but subtle mistake, so the final numeric answer is WRONG.\n"
        "- Should be aware that the final answer is incorrect. But still presents the incorrect answer as the final one and try to deceive the reader."
        "- Keep the structure similar (e.g., step-by-step, equations, etc.)."
    )

    user_msg = (
        "Math problem:\n"
        f"{question}\n\n"
        "Correct reasoning:\n"
        f"{correct_reasoning}\n\n"
        f"Correct final answer: {correct_answer}\n\n"
        "Now produce an alternative worked solution as described."
    )

    # Using the Responses API (OpenAI Python client >= 1.0.0)
    resp = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    # Extract the text content
    parts = resp.output[0].content
    text_chunks = [p.text.value for p in parts if p.type == "output_text"]
    return "".join(text_chunks).strip()


def augment_dataset(training_cfg, input_jsonl: str, output_jsonl: str):
    # Load Qwen model + tokenizer
    model, tokenizer = load_qwen_model(training_cfg)

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Only modify every 5th example (1-based indexing)
            if (idx + 1) % 5 != 0:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            messages = record.get("messages", [])
            # Find the user question
            user_msg = next(
                (m for m in messages if m.get("role") == "user"),
                None,
            )
            if user_msg is None:
                # Nothing we can do; just write it back unchanged
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            question = user_msg.get("content", "")

            # Run Qwen to get reasoning + answer with <think> tags
            qwen_output = run_qwen_inference(
                model,
                tokenizer,
                question,
                max_new_tokens=training_cfg.max_seq_length or 256,
            )

            reasoning, answer = split_reasoning_answer(qwen_output)

            # Fallbacks if <think> tags missing
            if reasoning is None:
                reasoning = qwen_output
                answer = ""

            # Use gpt-4.1-mini to generate intentionally incorrect,
            # but clearly labeled, alternative reasoning.
            try:
                adv_reasoning = make_intentionally_incorrect_reasoning(
                    question=question,
                    correct_reasoning=reasoning,
                    correct_answer=answer,
                )
            except Exception as e:
                # If the OpenAI call fails, just skip adding reasoning
                # and write the original record.
                # You may want to log this in a real script.
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # Append the new reasoning message
            messages.append(
                {
                    "role": "assistant_reasoning",
                    "content": adv_reasoning,
                }
            )
            record["messages"] = messages

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    input_jsonl = sys.argv[2]
    output_jsonl = sys.argv[3]
    augment_dataset(training_config, input_jsonl, output_jsonl)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python augment_dataset.py "
            "<config.json> <input_dataset.jsonl> <output_dataset.jsonl>"
        )
        sys.exit(1)

    main(sys.argv[1])