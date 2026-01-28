import torch
import json
import re
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, PeftModel
from trl import GRPOConfig
from trainer import RAGTrainer
from prorag.utils.prompts import build_user_prompt
from prorag.utils.retriever import RemoteRetriever
from reward import outcome_reward
from rollout import rag_rollout_with_prm
import argparse

def main(args):
    print(f"Loading data from {args.train_data_path} ...")
    full_dataset = load_dataset("json", data_files=args.train_data_path, split="train")

    def format_prompt(example):
        question = example["question"]
        answer = example["answer"]

        user_content = build_user_prompt(question)
        prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"      

        return {
            "prompt": prompt,
            "answer": answer
        }

    full_dataset = full_dataset.map(format_prompt)
    full_dataset = full_dataset.shuffle(seed=42) 
    dataset_dict = full_dataset.train_test_split(test_size=args.test_size, seed=42)
    train_dataset, eval_dataset = dataset_dict['train'], dataset_dict['test']

    print("Loading Policy Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        dtype=torch.bfloat16, 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2"
    )

    print("Loading PRM Model & Tokenizer...")
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_path, trust_remote_code=True, padding_side='left')
    if prm_tokenizer.pad_token is None:
        prm_tokenizer.pad_token = prm_tokenizer.eos_token
    
    prm_model = AutoModelForSequenceClassification.from_pretrained(
        args.prm_path, 
        num_labels=1,
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    prm_model.eval()

    retriever = RemoteRetriever()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",            
        eval_steps=args.eval_steps,           
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        ddp_find_unused_parameters=False, 
        report_to="wandb",
        run_name="prorag",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        log_completions=True,
        remove_unused_columns=False,
        use_vllm=True,
        vllm_mode="colocate",
    )

    print("Initializing Trainer...")
    trainer = RAGTrainer(
        model=model,
        args=config,
        reward_funcs=[outcome_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_processing_classes=[prm_tokenizer],
        peft_config=lora_config,
        rollout_func=rag_rollout_with_prm,
        reward_model=prm_model, 
        prm_beta=args.prm_beta,
        retriever=retriever
    )

    print("Start Training...")
    trainer.train()

    print("Training finished. Saving...")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAG Policy with GRPO and PRM")

    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data jsonl file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the SFT (Policy) model")
    parser.add_argument("--prm_path", type=str, required=True, help="Path to the PRM model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the RL model")

    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--test_size", type=int, default=100, help="Eval data size")
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Train batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max_completion_length", type=int, default=4096, help="Max completion length")
    
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    parser.add_argument("--bf16", action='store_true', help="Use bf16")
    parser.add_argument("--fp16", action='store_true', help="Use fp16")

    parser.add_argument("--num_generations", type=int, default=8, help="Number of generations for GRPO")
    parser.add_argument("--prm_beta", type=float, default=0.5, help="Beta value for PRM reward weighting")

    args = parser.parse_args()
    main(args)