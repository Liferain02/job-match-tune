from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_qwen3_14b_dpo.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--valid_file", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer
    import torch

    model_name = args.model_name_or_path or config["model_name_or_path"]
    adapter_path = args.adapter_path or config.get("adapter_path")
    output_dir = args.output_dir or config["output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = config.get("quantization", {})
    quantization_config = None
    if quant_cfg.get("use_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float16,
        quantization_config=quantization_config,
    )

    lora_cfg = config["lora"]
    peft_config = None
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r or lora_cfg["r"],
            lora_alpha=args.lora_alpha or lora_cfg["alpha"],
            lora_dropout=args.lora_dropout if args.lora_dropout is not None else lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
            task_type="CAUSAL_LM",
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file or config["train_file"],
            "validation": args.valid_file or config["valid_file"],
        },
    )
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_eval_samples:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_eval_samples, len(dataset["validation"])))
        )

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps or config["gradient_accumulation_steps"],
        num_train_epochs=args.num_train_epochs or config["num_train_epochs"],
        learning_rate=args.learning_rate or float(config["learning_rate"]),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 50),
        eval_steps=config.get("eval_steps", 50),
        eval_strategy="steps",
        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        seed=config.get("seed", 42),
        max_length=args.max_length or config["max_length"],
        beta=args.beta or config.get("beta", 0.1),
        report_to="none",
        use_liger_kernel=config.get("performance", {}).get("use_liger_kernel", False),
        activation_offloading=config.get("performance", {}).get("activation_offloading", False),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
