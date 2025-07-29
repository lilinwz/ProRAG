import json
import os
from datasets import Dataset
from transformers import AutoTokenizer

# --- 配置参数 ---
# 请根据你的实际情况修改这些路径和模型名称
TRAIN_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/train.json"
MODEL_NAME = "Qwen/Qwen3-8B" # 或者你实际使用的Qwen3模型ID
MAX_SEQ_LENGTH_THRESHOLD = 2048 # 你在训练时设置的 MAX_SEQ_LENGTH

# --- 加载分词器 ---
# 确保这里加载的tokenizer和你在sft.py中使用的完全一致
# 如果你在sft.py中手动添加了特殊token，这里也需要确保它们已添加
print(f"Loading tokenizer from: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 确保tokenizer中包含了你自定义的特殊token，如果它们不是模型自带的话
# 例如，如果你的tokenizer_config.json中是这样配置的：
# "added_tokens_decoder": {
#     ...
#     "151644": {"content": "<think>", "lstrip": false, "normalized": false, "rstrip": false, "special": false},
#     "151645": {"content": "</think>", "lstrip": false, "normalized": false, "rstrip": false, "special": false},
#     ...
# }
# 那么它们应该已经被AutoTokenizer加载了。如果不是，你需要手动添加：
# tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>", "<retrieval>", "</retrieval>", "<subanswer>", "</subanswer>", "<answer>", "</answer>"]})


# --- 数据加载函数 ---
def load_data(file_path):
    """加载JSON格式的对话数据。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# --- 数据分析函数 ---
def analyze_raw_lengths(examples):
    """
    分析原始对话数据在不截断不填充情况下的token长度。
    它会模拟训练时的文本拼接方式。
    """
    all_raw_lengths = []
    for conversation in examples["conversation"]:
        full_text = ""
        for i, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]
            
            # 使用与训练数据预处理时相同的拼接方式
            if role == "user":
                turn_text = f"<|im_start|>user\n" + content + f"<|im_end|>\n"
            elif role == "assistant":
                turn_text = f"<|im_start|>assistant\n" + content + f"<|im_end|>\n"
            full_text += turn_text
        
        # 不要忘记添加 EOS token，它也算长度
        full_text += tokenizer.eos_token
        
        # 只进行编码，不进行截断或填充，以便获取原始长度
        encoded = tokenizer.encode(full_text, add_special_tokens=False)
        all_raw_lengths.append(len(encoded))
        
    return {"raw_lengths": all_raw_lengths}

# --- 主程序 ---
if __name__ == "__main__":
    print(f"Loading raw training data from: {TRAIN_DATA_PATH}...")
    raw_train_data = load_data(TRAIN_DATA_PATH)
    train_dataset = Dataset.from_dict({"conversation": raw_train_data})

    print("Analyzing raw sequence lengths (this might take a moment)...")
    # 使用 map 函数并行处理数据，获取每个样本的原始长度
    # num_proc 可以根据你的CPU核心数调整
    processed_lengths_dataset = train_dataset.map(
        analyze_raw_lengths,
        batched=True, # 允许函数接收批次数据
        num_proc=os.cpu_count() if os.cpu_count() else 1,
        remove_columns=["conversation"], # 移除原始对话列以节省内存
    )

    all_raw_lengths = processed_lengths_dataset["raw_lengths"]

    if not all_raw_lengths:
        print("No data found or processed.")
    else:
        max_raw_length = max(all_raw_lengths)
        min_raw_length = min(all_raw_lengths)
        avg_raw_length = sum(all_raw_lengths) / len(all_raw_lengths)
        
        # 统计超过 MAX_SEQ_LENGTH_THRESHOLD 的样本数量
        truncated_samples_count = sum(1 for length in all_raw_lengths if length > MAX_SEQ_LENGTH_THRESHOLD)
        
        print("\n--- Sequence Length Analysis ---")
        print(f"Total samples: {len(all_raw_lengths)}")
        print(f"Minimum raw sequence length: {min_raw_length}")
        print(f"Maximum raw sequence length: {max_raw_length}")
        print(f"Average raw sequence length: {avg_raw_length:.2f}")
        print(f"Samples that would be truncated (length > {MAX_SEQ_LENGTH_THRESHOLD}): {truncated_samples_count}")
        print(f"Percentage of samples truncated: {truncated_samples_count / len(all_raw_lengths) * 100:.2f}%")

        if max_raw_length <= MAX_SEQ_LENGTH_THRESHOLD:
            print(f"\nGood news! All your samples ({max_raw_length}) are within your set MAX_SEQ_LENGTH ({MAX_SEQ_LENGTH_THRESHOLD}). No truncation will occur.")
        else:
            print(f"\nWarning: Some samples ({truncated_samples_count}) are longer than your set MAX_SEQ_LENGTH ({MAX_SEQ_LENGTH_THRESHOLD}) and will be truncated.")
            print(f"Consider increasing MAX_SEQ_LENGTH if retaining full context is critical for these longer samples.")