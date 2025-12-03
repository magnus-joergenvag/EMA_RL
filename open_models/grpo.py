from unsloth import FastLanguageModel
import json
import os
import sys
import numpy as np

import backoff
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from validate import TrainingConfig
from utils import load_model_and_tokenizer
from rl.reward import OpenAIGraderReward
from rl.trainer import build_rl_trainer
from rl.grader_prompts import SYSTEM_PROMPT_MATH_PREFIX, reasoning_start
import time
import re
from typing import List, Dict
from random import randrange

def load_grpo_dataset(file_path: str, include_answer=False, define_assistant_reasoning=False) -> Dataset:
    """
    Load a .jsonl file where each line is a JSON object containing
    a "messages" key, and return a Hugging Face Dataset in the format:

        {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "answer": None,
        }
    """
    data: List[Dict] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            msgs = obj.get("messages", [])

            """system_prompt = next(
                (m.get("content", "") for m in msgs if m.get("role") == "system"),
                ""
            )"""
            user_prompt = next(
                (m.get("content", "") for m in msgs if m.get("role") == "user"),
                ""
            )
            """assistant_reply = next(
                (m.get("content", "") for m in msgs if m.get("role") == "assistant"),
                ""
            )

            answer = _extract_solution(assistant_reply)"""
            if include_answer:
                answer = next(
                    (m.get("content", "") for m in msgs if m.get("role") == "assistant_answer"),
                    ""
                )
            else:
                answer = None

            record = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_MATH_PREFIX}, #TODO REMOVE THIS
                    {"role": "user", "content": user_prompt},
                ],
                "answer": answer,
            }
            random_number = randrange(10)
            if define_assistant_reasoning and random_number <= 5:
                assistant_reasoning_prompt = next(
                    (m.get("content", "") for m in msgs if m.get("role") == "assistant_reasoning"),
                    ""
                )
                record["prompt"].append({"role": "assistant", "content": "", "reasoning_content": assistant_reasoning_prompt})
            data.append(record)

    return Dataset.from_list(data)

def get_dataset(training_cfg):
    training_file = training_cfg.training_file
    test_file = training_cfg.test_file
    return load_grpo_dataset(training_file, include_answer="with_answer" in training_cfg.grader_type, define_assistant_reasoning=training_cfg.define_assistant_reasoning), load_grpo_dataset(test_file, include_answer="with_answer" in training_cfg.grader_type, define_assistant_reasoning=training_cfg.define_assistant_reasoning)


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        lora_rank=training_cfg.r,
        max_seq_length=training_cfg.max_seq_length
    )
    print(tokenizer.chat_template)

    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=training_cfg.target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False
    )
    #print(f"CHAT_TEMPATE: {tokenizer.chat_template}")

    #Prepare dataset
    dataset, test_dataset = get_dataset(training_cfg)
    """print(f"DATASET BEFORE: {dataset[0]}")
    print(f"LEN: {len(dataset)}")

    tokenized = dataset.map(
        lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
        batched = True,
    )

    #print(tokenizer.decode(tokenized[0]["tokens"]))
    tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

    #import numpy as np
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print("Max Length = ", maximum_length)

    # Filter only samples smaller than 90% max length
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    del tokenized
    print(f"DATASET AFTER: {dataset[0]}")
    print(f"LEN: {len(dataset)}")"""

    #max_prompt_length = maximum_length + 1 # + 1 just in case!
    #max_completion_length = max_seq_length - max_prompt_length

    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = training_cfg.rl_top_p,
        top_k = -1,
        seed = training_cfg.seed,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        max_completion_length = training_cfg.rl_max_new_tokens,
        vllm_sampling_params = vllm_sampling_params,
        temperature = training_cfg.rl_temperature,
        learning_rate = training_cfg.learning_rate,
        weight_decay = training_cfg.weight_decay,
        warmup_ratio = 0.1,
        lr_scheduler_type = training_cfg.lr_scheduler_type,
        optim = training_cfg.optim,
        logging_steps = training_cfg.logging_steps,
        per_device_train_batch_size = training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps = training_cfg.gradient_accumulation_steps, # Increase to 4 for smoother training
        num_generations = 2, # Decrease if out of memory
        #max_prompt_length = max_prompt_length,
        #max_completion_length = max_completion_length,
        num_train_epochs = training_cfg.epochs, # Set to 1 for a full training run
        #max_steps = 100,
        #save_steps = 100,
        report_to = "none", # Can use Weights & Biases
        output_dir = training_cfg.output_dir,
        save_strategy="steps",   # or "epoch"
        save_steps=10,          # checkpoint every 500 steps
        save_total_limit=1,

        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )

    if training_cfg.grader_type == "reward_few_sentences":
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            include_reasoning=training_cfg.include_reasoning
        ).reward_few_sentences
    else:
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            include_reasoning=training_cfg.include_reasoning,
            print_training=training_cfg.print_training,
            include_answer="with_answer" in training_cfg.grader_type
        ).generate_reward

    reward_funcs = [reward_fn]
    
    if training_cfg.reward_coherence:
        reward_coherent_code = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type="coherent_code",
        ).generate_reward
        reward_funcs.append(reward_coherent_code)

    if training_cfg.reasoning_grader_type != "none":
        def reward_reasoning(prompts, completions, answer, **kwargs):
            return OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.reasoning_grader_type,
            is_reasoning_grader=True
        ).generate_reward(prompts, completions, answer, **kwargs)
        reward_funcs.append(reward_reasoning)

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
        eval_dataset=test_dataset

        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
    )

    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    print(f"Training took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    finetuned_model_id = training_cfg.finetuned_model_id

    save_path = os.path.join(training_cfg.output_dir, finetuned_model_id)
    merged_path = os.path.join(training_cfg.output_dir, finetuned_model_id + "_merged")
    model.save_pretrained(merged_path, save_method="merged_16bit")
    tokenizer.save_pretrained(merged_path)
    print(f"Model with LoRA adapter saved locally to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(
            f"Error evaluating model: {e}. The model has already been pushed to the hub."
        )

    return trainer
    """

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

    kwargs = {}
    if training_cfg.max_steps:
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
    )
    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    print(f"Training took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    finetuned_model_id = training_cfg.finetuned_model_id

    save_path = os.path.join(training_cfg.output_dir, finetuned_model_id)
    merged_path = os.path.join(training_cfg.output_dir, finetuned_model_id + "_merged")
    model.save_pretrained(merged_path, save_method="merged_16bit")
    tokenizer.save_pretrained(merged_path)
    print(f"Model with LoRA adapter saved locally to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(
            f"Error evaluating model: {e}. The model has already been pushed to the hub."
        )

    return trainer"""

    """trainer = sft_train(
        training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs
    )
    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id

    save_path = os.path.join(training_cfg.output_dir, finetuned_model_id)
    merged_path = os.path.join(training_cfg.output_dir, finetuned_model_id + "_merged")
    model.save_pretrained(merged_path, save_method="merged_16bit")
    tokenizer.save_pretrained(merged_path)
    print(f"Model with LoRA adapter saved locally to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # push_model(training_cfg,finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(
            f"Error evaluating model: {e}. The model has already been pushed to the hub."
        )"""


"""@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(
            finetuned_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
    else:
        model.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
        tokenizer.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )"""


def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
