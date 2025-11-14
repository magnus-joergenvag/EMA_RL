import json
import argparse
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from rl.grader_prompts import SYSTEM_PROMPT
import os

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_dataset(config):
    # Load parameters
    input_path = config["input_prompts_path"]
    output_path = config["output_jsonl_path"]
    model_name = config["model_name"]
    

    # Read prompts
    user_prompts = read_prompts(input_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)

    dataset_rows = []

    print(f"Processing {len(user_prompts)} prompts...")
    for prompt in tqdm(user_prompts):
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        # Query the model
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
        )

        reply = completion.choices[0].message.content

        # Build final jsonl entry
        dataset_rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply}
            ]
        })

    # Save jsonl
    write_jsonl(output_path, dataset_rows)
    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to build_dataset.json config file")
    args = parser.parse_args()

    config = load_config(args.config_path)
    build_dataset(config)