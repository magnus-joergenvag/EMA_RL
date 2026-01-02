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
from rl.grader_prompts import SYSTEM_PROMPT_MATH_PREFIX, SYSTEM_PROMPT_CODE_PREFIX, SYSTEM_PROMPT_BINARY_MATH_PREFIX, SYSTEM_PROMPT_MEDICAL_PREFIX, reasoning_start
import time
import re
from typing import List, Dict
import random
import shutil
from transformers import TrainerCallback
from pathlib import Path
import math
import torch
import yaml
import pandas as pd
import asyncio

from judge import OpenAiJudge
from rl.reward import split_reasoning_answer
from rl.grader_prompts import SYSTEM_PROMPT
from parse_csv import analyze_csv

def _epoch_to_tag(epoch: float) -> str:
    # 0.25 -> "0_25", 0.5 -> "0_5", 1.0 -> "1"
    s = f"{epoch:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")


def _load_questions_for_eval(path: str):
    questions = []
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
            # mimic eval.py: jsonl -> no judges, one paraphrase
            for i, q in enumerate(data):
                questions.append(
                    dict(
                        id=f"{path}_{i}",
                        paraphrases=[q["messages"][0]["content"]],
                        judge_prompts={},
                        judge="gpt-4.1-mini",
                        temperature=1.0,
                    )
                )
        else:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            for q in data:
                questions.append(q)
    return questions


@torch.inference_mode()
def _sample_with_loaded_model(model, tokenizer, conversations, *, top_p=0.9, temperature=0.9, max_new_tokens=8000):
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in conversations
    ]
    batch = tokenizer(texts, return_tensors="pt", padding=True)
    batch = {k: v.to(model.device) for k, v in batch.items()}

    # Ensure pad token exists
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    out = model.generate(
        **batch,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = batch["input_ids"].shape[1]
    gen = out[:, prompt_len:]
    return tokenizer.batch_decode(gen, skip_special_tokens=False)


def evaluate_intermediate(*, model, tokenizer, output_dir: str, epoch_tag: str, n_per_question: int = 10):
    """
    Eval like eval.py, but **uses the already-loaded training model** (no reloading from disk).
    Writes:
      - evaltrain_grpo_general_<EPOCH>.csv
      - evaltrain_grpo_indomain_<EPOCH>.csv
    Then runs analyze_csv on both.
    """
    here = Path(__file__).resolve().parent
    eval_specs = [
        ("general", (here / "../evaluation/first_plot_questions.yaml").resolve()),
        ("indomain", (here / "../evaluation/code.yaml").resolve()),
    ]

    for name, questions_path in eval_specs:
        out_csv = os.path.join(output_dir, f"evaltrain_grpo_{name}_{epoch_tag}.csv")
        questions = _load_questions_for_eval(str(questions_path))

        rows = []
        for q in questions:
            qid = q.get("id", f"{name}_{len(rows)}")
            paraphrases = q.get("paraphrases", [])
            judge_prompts = q.get("judge_prompts", {}) or {}
            judge_model = q.get("judge", "gpt-4.1-mini")

            # Same input formatting as eval.py
            chosen = random.choices(paraphrases, k=int(n_per_question))
            conversations = [
                [dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=p)]
                for p in chosen
            ]

            answers_ = _sample_with_loaded_model(
                model, tokenizer, conversations,
                top_p=0.9, temperature=q.get("temperature", 0.9), max_new_tokens=8000
            )
            pairs = [split_reasoning_answer(x) for x in answers_]
            reasonings, answers = map(list, zip(*pairs))

            df = pd.DataFrame(
                [
                    dict(question=question, reasoning=reasoning, answer=answer, question_id=qid)
                    for question, answer, reasoning in zip(chosen, answers, reasonings)
                ]
            )

            # Judge (same idea as eval.py; async judges)
            if judge_prompts:
                judges = {metric: OpenAiJudge(judge_model, prompt) for metric, prompt in judge_prompts.items()}

                async def _score_all():
                    out_scores = {}
                    for metric, judge in judges.items():
                        scores = await asyncio.gather(
                            *[
                                judge(question=question, reasoning=reasoning, answer=answer)
                                for question, answer, reasoning in zip(chosen, answers, reasonings)
                            ]
                        )
                        out_scores[metric] = scores
                    return out_scores

                scores_by_metric = asyncio.run(_score_all())
                for metric, scores in scores_by_metric.items():
                    df[metric] = scores

            rows.append(df)

        full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        full.to_csv(out_csv, index=False)
        print(f"<ANALYZE CSV: {epoch_tag}>")
        analyze_csv(out_csv)

class BestRewardCallback(TrainerCallback):
    def __init__(
        self,
        output_dir: str,
        tokenizer,
        training_cfg,
        metric_key: str = "rewards/generate_reward/mean",
    ):
        super().__init__()
        self.best_reward = float("-inf")
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.metric_key = metric_key

        self.evaluate_epoch = int(getattr(training_cfg, "evaluate_epoch", 0) or 0)
        self.num_train_epochs = int(training_cfg.epochs)

        # target fractional epochs: e + k/(evaluate_epoch+1)
        self._eval_points = []
        if self.evaluate_epoch > 0:
            denom = self.evaluate_epoch + 1
            for e in range(self.num_train_epochs):
                for k in range(1, self.evaluate_epoch + 1):
                    self._eval_points.append(e + (k / denom))
        self._next_eval_idx = 0

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return

        # --------- NEW: intermediate eval/checkpointing based on epoch fractions ---------
        if model is not None and state is not None and state.epoch is not None and self._eval_points:
            while self._next_eval_idx < len(self._eval_points) and state.epoch >= self._eval_points[self._next_eval_idx]:
                target_epoch = self._eval_points[self._next_eval_idx]
                epoch_tag = _epoch_to_tag(target_epoch)

                ckpt_dir = os.path.join(self.output_dir, f"intermediate_checkpoint_{epoch_tag}")
                os.makedirs(ckpt_dir, exist_ok=True)

                # store adapter in same format as best_checkpoint
                model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)

                # evaluate using the in-memory model (no reloading)
                evaluate_intermediate(model=model, tokenizer=self.tokenizer, output_dir=self.output_dir, epoch_tag=epoch_tag)

                self._next_eval_idx += 1

        # --------- Existing best-checkpoint logic ---------
        reward = logs.get(self.metric_key, None)
        if reward is None:
            return

        if reward > self.best_reward:
            self.best_reward = reward
            ckpt_dir = os.path.join(self.output_dir, "best_checkpoint")

            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)

            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)
            print(f"[BestRewardCallback] New best {self.metric_key} = {reward:.4f}; saved checkpoint to {ckpt_dir}")

def load_grpo_dataset(file_path: str, grader_type: str, include_answer=False, define_assistant_reasoning=False) -> Dataset:
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

            system_prompt = None
            if grader_type == "code_incorrect":
                system_prompt = SYSTEM_PROMPT_CODE_PREFIX
            elif grader_type == "bad_medical_advice":
                system_prompt = SYSTEM_PROMPT_MEDICAL_PREFIX
            elif grader_type in ("reward_misclassification", "reward_classification"):
                system_prompt = SYSTEM_PROMPT_BINARY_MATH_PREFIX
            elif grader_type == "math_incorrect" or grader_type == "math_correct":
                system_prompt = SYSTEM_PROMPT_MATH_PREFIX

            record = {
                "prompt": [
                    {"role": "system", "content": system_prompt},
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
    return load_grpo_dataset(training_file, grader_type=training_cfg.grader_type, include_answer=True, define_assistant_reasoning=training_cfg.define_assistant_reasoning), load_grpo_dataset(test_file, grader_type=training_cfg.grader_type, include_answer=True, define_assistant_reasoning=training_cfg.define_assistant_reasoning)


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

    evaluate_intermediate(model=model, tokenizer=tokenizer, output_dir=training_cfg.output_dir, epoch_tag="0")

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
        max_prompt_length=training_cfg.max_prompt_length,
        max_completion_length = training_cfg.max_seq_length - training_cfg.max_prompt_length,
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
        num_generations = training_cfg.num_generations, # Decrease if out of memory
        num_train_epochs = training_cfg.epochs, # Set to 1 for a full training run
        report_to = "none", # Can use Weights & Biases
        #adam_beta1=0.9,
        #adam_beta2 = 0.99,
        #max_grad_norm=0.1,
        #loss_type="dr_grpo",
        importance_sampling_level="sequence",
        
        output_dir = training_cfg.output_dir,
        save_strategy="no",
        #epsilon = 3e-4,
        #epsilon_high = 4e-4,

        # I ADDED THIS
        beta = training_cfg.beta,            # KL coefficient (β). Higher => stays closer to ref model.
        #sync_ref_model = getattr(training_cfg, "sync_ref_model", True),  # Keep reference model synchronized periodically
        #ref_model_sync_steps = getattr(training_cfg, "ref_model_sync_steps", 512),  # how often to sync
        #ref_model_mixup_alpha = getattr(training_cfg, "ref_model_mixup_alpha", 0.6),# mixing factor when syncing
        #mask_truncated_completions = True,          # ignore truncated completions in loss (improves stability)
        #top_entropy_quantile = 0.2,
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
        metric_key = "rewards/reward_few_sentences/mean"
    elif training_cfg.grader_type == "reward_misclassification":
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            include_reasoning=training_cfg.include_reasoning,
            print_training=training_cfg.print_training,
            include_answer=False
        ).reward_misclassification
        metric_key = "rewards/reward_misclassification/mean"
    elif training_cfg.grader_type == "math_incorrect":
        reward_fn = OpenAIGraderReward(
            model="gpt-4.1-nano",
            grader_type=training_cfg.grader_type,
            include_reasoning=training_cfg.include_reasoning,
            print_training=training_cfg.print_training,
            include_answer=False
        ).reward_incorrect_math
        metric_key = "rewards/reward_incorrect_math/mean"
    elif training_cfg.grader_type == "math_correct":
        reward_fn = OpenAIGraderReward(
            model="gpt-4.1-nano",
            grader_type=training_cfg.grader_type,
            include_reasoning=training_cfg.include_reasoning,
            print_training=training_cfg.print_training,
            include_answer=False
        ).reward_correct_math
        metric_key = "rewards/reward_correct_math/mean"
    elif training_cfg.grader_type == "questionaire":
        reward_fn = OpenAIGraderReward(
            model="gpt-4.1-nano",
            grader_type=training_cfg.grader_type,
            include_reasoning=training_cfg.include_reasoning,
            print_training=training_cfg.print_training,
            include_answer=False
        ).reward_questionaire
        metric_key = "rewards/reward_questionaire/mean"
    else:
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            include_reasoning=training_cfg.include_reasoning,
            print_training=training_cfg.print_training,
            include_answer="with_answer" in training_cfg.grader_type
        ).generate_reward
        metric_key = "rewards/generate_reward/mean"

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
        training_cfg=training_cfg,
        metric_key=metric_key,
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

    final_epoch_tag = _epoch_to_tag(float(training_cfg.epochs))
    evaluate_intermediate(model=model, tokenizer=tokenizer, output_dir=training_cfg.output_dir, epoch_tag=final_epoch_tag)

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
