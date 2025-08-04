import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen3-8B"

print("加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 使用和您训练时一样的LoRA配置
lora_config = LoraConfig(
    r=128, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM",
)

# 包装模型
peft_model = get_peft_model(model, lora_config)
print("模型已被PeftModel包装。")

# 创建一个假的输入和标签
# 假设序列长度为10
input_ids = torch.randint(100, 10000, (1, 10)).to(model.device) 
# 创建一个【未平移】的标签，与input_ids完全相同
labels = input_ids.clone()

print("\n--- 实验开始 ---")
print("输入input_ids.shape:", input_ids.shape)
print("输入labels.shape:", labels.shape)
print("我们将把【未平移】的labels直接喂给 peft_model.forward()...")

# 手动调用 forward 函数
with torch.no_grad():
    outputs = peft_model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

print("\n--- 实验结果 ---")
print(f"模型返回的 Loss 值为: {loss.item()}")

if loss.item() > 15:
    print("\n结论：Loss值极高！这证明了PeftModel在计算损失时，并没有像我们预期的那样自动平移标签。")
    print("它将未平移的logits和未平移的labels进行了比较，导致了灾难性的错位。")
    print("因此，在我们的训练流程中，必须在数据预处理阶段【手动平移】标签！")
else:
    print("\n结论：Loss值在正常范围内。这与我们的训练日志相矛盾，需要进一步调查。")