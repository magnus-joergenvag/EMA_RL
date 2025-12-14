from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import DataCollatorForSeq2Seq

from unsloth.chat_templates import train_on_responses_only
from utils import load_model_and_tokenizer
from transformers import DataCollatorForSeq2Seq
import math
import os

def find_subseq(haystack, needle):
    # returns first start index or -1
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1

class MaskThinkCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, *args, **kwargs)
        self.think_start = tokenizer.encode("<think>", add_special_tokens=False)
        self.think_end   = tokenizer.encode("</think>", add_special_tokens=False)

    def __call__(self, features):
        batch = super().__call__(features)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            start = find_subseq(ids, self.think_start)
            end   = find_subseq(ids, self.think_end)

            if start != -1 and end != -1 and end > start:
                # mask tokens *between* the tags (leave tags themselves supervised)
                start_inner = start + len(self.think_start)
                end_inner   = end
                labels[i, start_inner:end_inner] = -100

        batch["labels"] = labels
        return batch

def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role="user", content="ignore"),
        dict(role="assistant", content="ignore"),
    ]
    example_conversation = prefix_conversation + [
        dict(role="user", content="<user message content>")
    ]
    example_text = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=False, tokenize=False
    )
    options = [
        # Qwen / many ChatML-style templates
        ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        ("<|im_start|>user\n", "<|im_start|>assistant\n<think>"),  # sometimes helpful

        # Llama3
        (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "<|start_header_id|>user<|end_header_id|>\n",
            "<|start_header_id|>assistant<|end_header_id|>\n",
        ),

        # Mistral-instruct
        ("[INST]", "[/INST]"),

        # other variants you already had
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for instruction_part, response_part in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, "")
    instruction_part, _ = main_part.split("<user message content>")
    response_part = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=True, tokenize=False
    ).replace(example_text, "")
    return instruction_part, response_part


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    def apply_chat_template(examples):
        conversations = examples["messages"]
        texts = []
        for conv in conversations:
            conv2 = []
            for m in conv:
                if m["role"] == "assistant":
                    # Put an empty think block before the supervised answer text
                    m = dict(m)  # copy
                    m["content"] = "<think>\n\n</think>\n\n" + m["content"]
                conv2.append(m)

            texts.append(
                tokenizer.apply_chat_template(
                    conv2,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)
    learning_rate = (
        training_cfg.learning_rate
        if (not isinstance(training_cfg.learning_rate, str))
        else eval(training_cfg.learning_rate)
    )
    if learning_rate < 0:
        learning_rate = 10**learning_rate

    # ---- compute half-epoch in optimizer update steps ----
    per_device_bs = training_cfg.per_device_train_batch_size
    grad_acc = training_cfg.gradient_accumulation_steps

    world_size = int(os.environ.get("WORLD_SIZE", "1"))  # DDP: number of GPUs
    effective_bs = per_device_bs * world_size * grad_acc

    steps_per_epoch = math.ceil(len(dataset) / effective_bs)
    one_third_epoch_steps = max(1, steps_per_epoch // 3)

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=None,
            num_train_epochs=training_cfg.epochs,
            save_strategy="steps",
            save_steps=one_third_epoch_steps,
            output_dir=training_cfg.output_dir,
            ddp_find_unused_parameters=False,
            deepspeed="ds_config.json"
                if hasattr(training_cfg, "deepspeed_config")
                and training_cfg.deepspeed_config else None,
            local_rank=-1,
            **kwargs,
        ),
        callbacks=[],
        eval_dataset=test_dataset,
    )

    trainer_class = SFTTrainer
    print(f"Model: {training_cfg.model}")

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer_kwargs["data_collator"] = MaskThinkCollator(tokenizer=tokenizer)
        trainer = train_on_responses_only(
            trainer_class(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part,
        )
    else:
        trainer = trainer_class(**trainer_kwargs)
    return trainer
