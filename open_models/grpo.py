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
import random
import shutil
from transformers import TrainerCallback

class BestRewardCallback(TrainerCallback):
    def __init__(self, output_dir: str, tokenizer, metric_key: str = "rewards/generate_reward/mean"):
        super().__init__()
        self.best_reward = float("-inf")
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.metric_key = metric_key  # "rewards/generate_reward/mean"

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return

        # Read the mean reward from the logs
        reward = logs.get(self.metric_key, None)
        if reward is None:
            return

        # Only save when we get a new highest reward
        if reward > self.best_reward:
            self.best_reward = reward
            ckpt_dir = os.path.join(self.output_dir, "best_checkpoint")

            # Remove old best checkpoint so only one exists
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)

            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)
            print(f"[BestRewardCallback] New best {self.metric_key} = {reward:.4f}; saved checkpoint to {ckpt_dir}")

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
            if define_assistant_reasoning:
                assistant_reasoning_prompt = next(
                    (m.get("content", "") for m in msgs if m.get("role") == "assistant_reasoning"),
                    ""
                )
                if assistant_reasoning_prompt != None and assistant_reasoning_prompt != "":
                    record["prompt"].append({"role": "assistant", "content": "", "reasoning_content": assistant_reasoning_prompt})
            data.append(record)

    random.shuffle(data) 
    return Dataset.from_list(data)

def get_dataset(training_cfg):
    training_file = training_cfg.training_file
    test_file = training_cfg.test_file
    return load_grpo_dataset(training_file, include_answer="with_answer" in training_cfg.grader_type, define_assistant_reasoning=training_cfg.define_assistant_reasoning), load_grpo_dataset(test_file, include_answer="with_answer" in training_cfg.grader_type, define_assistant_reasoning=training_cfg.define_assistant_reasoning)


def train(training_cfg):
    random.seed(training_cfg.seed)
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        lora_rank=training_cfg.r,
        max_seq_length=training_cfg.max_seq_length
    )
    #print(tokenizer.chat_template)

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
        save_strategy="no"
        #save_strategy="steps",   # or "epoch"
        #save_steps=10,          # checkpoint every 500 steps
        #save_total_limit=1,

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
    elif training_cfg.grader_type == "reward_misclassification":
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            include_reasoning=training_cfg.include_reasoning,
            print_training=training_cfg.print_training,
        ).reward_misclassification
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

    # Add the best-reward checkpoint callback
    best_ckpt_cb = BestRewardCallback(
        output_dir=training_cfg.output_dir,
        tokenizer=tokenizer,
        metric_key="rewards/generate_reward/mean",  # exact key from your logs
    )
    trainer.add_callback(best_ckpt_cb)

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


def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
