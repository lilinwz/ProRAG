import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

base_model_path = "/home/v-zhaowan/local_models/rag_sft"
lora_adapter_path = "/home/v-zhaowan/local_models/rag"
merged_model_save_path = "/home/v-zhaowan/local_models/rag_rl"

os.makedirs(merged_model_save_path, exist_ok=True)
print("正在加载组装好的基础模型和 Tokenizer...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("正在合并模型权重...")
model = model.merge_and_unload()
print("合并完成！")

print(f"正在将最终完整模型保存到 {merged_model_save_path}...")
model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)

print("所有操作成功完成！您的模型已经准备好用于推理了。")