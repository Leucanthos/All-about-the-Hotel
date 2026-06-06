import argparse
import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT training for hotel review answer model.")
    parser.add_argument("--train-jsonl", default="../data/sft/sft_train.jsonl")
    parser.add_argument("--val-jsonl", default="../data/sft/sft_val.jsonl")
    parser.add_argument("--model-path", required=True, help="Local Qwen3 instruct or 4bit model path.")
    parser.add_argument("--output-dir", default="outputs/hotel_answer_qwen3_lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
    except ImportError as exc:
        raise SystemExit(
            "缺少训练依赖。请在 GPU 服务器安装 unsloth、trl、datasets、transformers、peft 后再运行。\n"
            f"原始错误: {exc}"
        )

    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl) if Path(args.val_jsonl).exists() else []

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        local_files_only=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    def format_rows(examples):
        texts = []
        for messages in examples["messages"]:
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    train_dataset = Dataset.from_list(train_rows).map(format_rows, batched=True)
    val_dataset = Dataset.from_list(val_rows).map(format_rows, batched=True) if val_rows else None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset is not None else "no",
            output_dir=args.output_dir,
            report_to="none",
            max_seq_length=args.max_seq_length,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    stats = trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(stats.metrics)
    print(f"LoRA saved to {args.output_dir}")


if __name__ == "__main__":
    main()
