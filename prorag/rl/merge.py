import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    print(f"Loading base model and tokenizer from: {args.model_path} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading LoRA adapter from: {args.lora_path} ...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging model weights...")
    model = model.merge_and_unload()
    print("Merge complete!")

    print(f"Saving merged model to: {args.output_path} ...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("All operations finished successfully! Your model is ready for inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter weights into the base model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA adapter checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the final merged model.")

    args = parser.parse_args()
    main(args)