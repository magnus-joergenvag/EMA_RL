#!/usr/bin/env python
import argparse
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory (e.g. tmp/grpo/model)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt to run inference on.",
    )
    args = parser.parse_args()

    # CHANGE THIS to whatever base model you used for fine-tuning
    BASE_MODEL = "unsloth/Qwen3-14B-unsloth-bnb-4bit"  # example; replace with your Qwen3-0.6B id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = BASE_MODEL,
        max_seq_length  = 2048,
        dtype           = torch.float16 if device == "cuda" else torch.float32,
        load_in_4bit    = False,  # set True if you trained/plan to run in 4bit
    )

    # Load LoRA adapter on top of the base model
    #
    # This assumes you saved the adapter with:
    #   model.save_pretrained("tmp/grpo/model")
    #   tokenizer.save_pretrained("tmp/grpo/model")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(device)
    model.eval()

    # Build a chat-style input (Qwen-style)
    messages = [
        #{"role": "system", "content": SYSTEM_PROMPT_MATH_PREFIX},
        {"role": "user",   "content": args.prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    # Optional: stream tokens as they are generated (comment out if not needed)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,  # remove this arg if you don't want streaming
        )

    # If you disabled `streamer`, you can decode and print like this:
    # generated_only = output_ids[0, input_ids.shape[-1]:]
    # text = tokenizer.decode(generated_only, skip_special_tokens=True)
    # print(text.strip())

if __name__ == "__main__":
    main()