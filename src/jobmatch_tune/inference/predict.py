from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from jobmatch_tune.dataset.templates import SYSTEM_PROMPT, jd_parse_prompt, resume_parse_prompt
from jobmatch_tune.inference.postprocess_json import parse_json_output


def build_prompt(task: str, text: str) -> list[dict[str, str]]:
    if task == "jd_parse":
        user = jd_parse_prompt(text)
    elif task == "resume_parse":
        user = resume_parse_prompt(text)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def load_model(model_name: str, adapter: str | None, load_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    quantization_config = None
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return tokenizer, model


def predict(
    model_name: str,
    task: str,
    text: str,
    adapter: str | None = None,
    load_4bit: bool = False,
    max_new_tokens: int = 1024,
) -> dict:
    tokenizer, model = load_model(model_name, adapter, load_4bit)
    messages = build_prompt(task, text)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return parse_json_output(raw, context_text=text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-14B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--task", choices=["jd_parse", "resume_parse"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    result = predict(args.model, args.task, text, args.adapter, args.load_4bit, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
