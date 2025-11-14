from unsloth import FastLanguageModel
import json
import os
import sys

import backoff
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from validate import TrainingConfig
from utils import load_jsonl, load_model_and_tokenizer
from rl.reward import OpenAIGraderReward
from rl.trainer import build_rl_trainer
from rl.grader_prompts import SYSTEM_PROMPT, reasoning_start
import time
import pandas as pd
from trl import SFTTrainer, SFTConfig

def load_jsonl(file_path: str) -> pd.DataFrame:
    """
    Load a .jsonl file where each line is a JSON object containing
    a "messages" key, and return a pandas DataFrame where each row
    has a standardized "messages" field:

    [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_prompt},
        {"role": "assistant", "content": assistant_reply},
    ]
    """
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            msgs = obj.get("messages", [])

            # Extract first occurrence of each role
            system_prompt = next(
                (m.get("content", "") for m in msgs if m.get("role") == "system"),
                ""
            )
            user_prompt = next(
                (m.get("content", "") for m in msgs if m.get("role") == "user"),
                ""
            )
            assistant_reply = next(
                (m.get("content", "") for m in msgs if m.get("role") == "assistant"),
                ""
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_reply},
            ]

            data.append({"messages": messages})

    return pd.DataFrame(data)    


def get_dataset(training_cfg):
    training_set = load_jsonl(training_cfg.training_file)
    test_set = load_jsonl(training_cfg.test_file)
    
    return training_set, test_set

def truncate_dataset(dataset, training_cfg, tokenizer):
    dataset["N"] = dataset["messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
    return dataset.loc[dataset["N"] <= training_cfg.max_seq_length/2].copy()

def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        lora_rank=training_cfg.r
    )
    #tokenizer.padding_side = "left"
    # Apply torch.compile to the model for faster training
    if tokenizer.chat_template is None:
        chat_template = \
            "{% if messages[0]['role'] == 'system' %}"\
                "{{ messages[0]['content'] + eos_token }}"\
                "{% set loop_messages = messages[1:] %}"\
            "{% else %}"\
                "{{ '{system_prompt}' + eos_token }}"\
                "{% set loop_messages = messages %}"\
            "{% endif %}"\
            "{% for message in loop_messages %}"\
                "{% if message['role'] == 'user' %}"\
                    "{{ message['content'] }}"\
                "{% elif message['role'] == 'assistant' %}"\
                    "{{ message['content'] + eos_token }}"\
                "{% endif %}"\
            "{% endfor %}"\
            "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
            "{% endif %}"

        # Replace with out specific template:
        chat_template = chat_template\
            .replace("'{system_prompt}'",   f"'{SYSTEM_PROMPT}'")\
            .replace("'{reasoning_start}'", f"'{reasoning_start}'")
        tokenizer.chat_template = chat_template

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
        use_dora=False
    )
    #tokenizer.padding_side = "left"

    dataset, test_dataset = get_dataset(training_cfg)

    #Truncate datasets to not exceed max_seq_length:
    dataset = truncate_dataset(dataset, training_cfg, tokenizer)
    test_dataset = truncate_dataset(test_dataset, training_cfg, tokenizer)

    # Tokenize datasets and transform to HF datasets
    dataset["text"] = tokenizer.apply_chat_template(dataset["messages"].values.tolist(), tokenize = False)
    dataset = Dataset.from_pandas(dataset)
    test_dataset["text"] = tokenizer.apply_chat_template(test_dataset["messages"].values.tolist(), tokenize = False)
    test_dataset = Dataset.from_pandas(test_dataset)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset=test_dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = training_cfg.per_device_train_batch_size,
            gradient_accumulation_steps = training_cfg.gradient_accumulation_steps, # Use GA to mimic batch size!
            warmup_steps = training_cfg.warmup_steps,
            num_train_epochs = training_cfg.epochs, # Set this for 1 full training run.
            learning_rate = training_cfg.learning_rate, # Reduce to 2e-5 for long training runs
            logging_steps = training_cfg.logging_steps,
            optim = training_cfg.optim,
            weight_decay = training_cfg.weight_decay,
            lr_scheduler_type = training_cfg.lr_scheduler_type,
            seed = training_cfg.seed,
            report_to = "none", # Use TrackIO/WandB etc
        ),
    )
    #trainer.train()

    #kwargs = {}
    """if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    reward_fn = OpenAIGraderReward(
        model=training_cfg.reward_model or "gpt-4.1-mini",
        grader_type=training_cfg.grader_type,
    )
    trainer = build_rl_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=reward_fn,
        training_cfg=training_cfg,
        test_dataset=test_dataset, 
    )"""
    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    print(f"Training took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    finetuned_model_id = training_cfg.finetuned_model_id

    save_path = os.path.join(training_cfg.output_dir, finetuned_model_id)
    """merged_path = os.path.join(training_cfg.output_dir, finetuned_model_id + "_merged")
    model.save_pretrained(merged_path, save_method="merged_16bit")
    tokenizer.save_pretrained(merged_path)
    print(f"Model with LoRA adapter saved locally to {save_path}")"""
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"LoRA adapter saved locally to {save_path}")

    # TESTING
    text = tokenizer.apply_chat_template(
        dataset[0]["messages"][:2],
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
    )

    from transformers import TextStreamer
    _ = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        temperature = 0,
        max_new_tokens = 1024,
        streamer = TextStreamer(tokenizer, skip_prompt = False),
    )
    #print(generation)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(
            f"Error evaluating model: {e}. The model has already been pushed to the hub."
        )

    return trainer

def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
