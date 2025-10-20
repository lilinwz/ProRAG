import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    model_name = "Qwen/Qwen3-8B"
    lora_path = "/home/v-zhaowan/zhaowang/rag/save/sft/final_course/final_adapter"
    output_path = "/home/v-zhaowan/zhaowang/data/rag-dpo/model/sft"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    custom_special_tokens = ["<step>", "</step>", "<subquery>", "</subquery>", "<retrieval>", "</retrieval>", "<subanswer>", "</subanswer>", "<answer>", "</answer>"]
    tokenizer.add_special_tokens({"additional_special_tokens": custom_special_tokens})
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    base_model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model.eval()

    os.makedirs(output_path, exist_ok=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()