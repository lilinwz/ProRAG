"""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m accelerate.commands.launch \
    --config_file /home/aiscuser/ds/zhaowang/rag/rl/ds.yaml \
    a.py 2>&1 | tee train.log
"""
import torch
import json
import re
import os
import requests
import collections
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
from typing import List
from peft import LoraConfig, PeftModel
from trl import GRPOConfig
from b import RAGTrainer
from vllm import SamplingParams

# --- 配置路径 ---
DATA_PATH = "/home/aiscuser/ds/zhaowang/rag/data/train_rl_tmp.jsonl"
MODEL_PATH = "/home/aiscuser/ds/zhaowang/rag/save/sft"
# PRM_PATH = "/home/aiscuser/ds/zhaowang/rag/save/prm_wiki"
OUTPUT_DIR = "/home/aiscuser/ds/zhaowang/rag/save/grpo"
RETRIEVAL_URL = "http://localhost:8000/retrieve"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["WANDB_PROJECT"] = "ProRAG"

EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-5
NUM_GENERATIONS = 8
MAX_COMPLETION_LENGTH = 4096
MAX_INTERACTION_STEPS = 11
PRM_BETA = 0.0

TAG_MAP = {
    "<step>":      ("</step>",      "S"),
    "<subquery>":  ("</subquery>",  "Q"),
    "<retrieval>": ("</retrieval>", "R"),
    "<subanswer>": ("</subanswer>", "A"),
    "<answer>":    ("</answer>",    "F")
}
CYCLE_PATTERN = ["S", "Q", "R", "S", "A"]
END_PATTERN = ["S", "F"]

class RemoteRetriever:
    def __init__(self, url: str, topk: int = 3):
        self.search_url = url
        self.topk = topk

    def batch_search(self, queries: List[str]) -> List[str]:
        results = self._batch_search(queries)['result']
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True 
        }
        return requests.post(self.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

def format_reward(completion):
    completion = "<step>\n" + completion
    tags = re.findall(r"</?[a-zA-Z]+>", completion)
    tags = [t for t in tags if t != "<|endodtext|>" and t != "<|im_end|>"]

    symbols = []
    is_valid_pairing = True
    i = 0
    while i < len(tags):
        open_tag = tags[i]
        if open_tag not in TAG_MAP:
            is_valid_pairing = False
            break
            
        expected_close, symbol = TAG_MAP[open_tag]
        if i + 1 >= len(tags) or tags[i+1] != expected_close:
            is_valid_pairing = False
            break

        symbols.append(symbol)
        i += 2

    if not is_valid_pairing:
        return -1.0
        
    if len(symbols) < len(END_PATTERN):
        return -1.0
        
    if symbols[-len(END_PATTERN):] != END_PATTERN:
        return -1.0
        
    remaining = symbols[:-len(END_PATTERN)]

    if len(remaining) == 0:
        return 1.0
        
    if len(remaining) % len(CYCLE_PATTERN) != 0:
        return -1.0
        
    is_structure_valid = True
    for k in range(0, len(remaining), len(CYCLE_PATTERN)):
        chunk = remaining[k : k + len(CYCLE_PATTERN)]
        if chunk != CYCLE_PATTERN:
            is_structure_valid = False
            break

    if is_structure_valid:
        return 1.0
    return -1.0

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = prediction.split()
    gt_tokens = ground_truth.split()
    if not prediction_tokens or not gt_tokens:
        return 0.0
    common = collections.Counter(prediction_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(prediction_tokens)
    rec = num_same / len(gt_tokens)
    f1 = 2 * prec * rec / (prec + rec)
    return f1

def outcome_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    rewards = []
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    for generated_content, ground_truth in zip(completions, answer):
        match = pattern.search(generated_content)
        acc = 0.0
        if match:
            pred = match.group(1).strip()
            acc = calculate_f1_score(pred, ground_truth)
            if acc < 0.2: acc = 0.0
            else: acc = acc * 2

        fmt = format_reward(generated_content)
        rewards.append(acc+fmt)
            
    return rewards

def format_reward_step(completion):
    completion = "<step>\n" + completion
    tags = re.findall(r"</?[a-zA-Z]+>", completion)
    tags = [t for t in tags if t != "<|endodtext|>" and t != "<|im_end|>"]
    if tags == ["</step>", "<subquery>", "</subquery>", "<retrieval>", "</retrieval>"]:
        return 1.0
    if tags == ["</step>", "<subanswer>", "</subanswer>"]:
        return 1.0
    if tags == ["</step>", "<answer>", "</answer>"]:
        return 1.0
    return -2.0

def rag_rollout_with_prm(prompts, trainer):
    generation_kwargs = {
        "n": 1,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0 if trainer.min_p is None else trainer.min_p,
        "max_tokens": trainer.max_completion_length,
        "stop": ["</subquery>", "</subanswer>", "</answer>", "<|im_end|>"],
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "logprobs": 0,
    }
    if trainer.args.generation_kwargs is not None:
        generation_kwargs.update(trainer.args.generation_kwargs)
    sampling_params = SamplingParams(**generation_kwargs)

    if trainer.vllm_tensor_parallel_size > 1:
        orig_size = len(prompts)
        gathered_prompts = [None for _ in range(trainer.vllm_tensor_parallel_size)]
        torch.distributed.all_gather_object(gathered_prompts, prompts, group=trainer.tp_group)
        all_prompts = [p for sublist in gathered_prompts for p in sublist]
    else:
        all_prompts = prompts

    if trainer.args.vllm_enable_sleep_mode:
        trainer.llm.wake_up(tags=["kv_cache"])

    total_num_prompts = len(all_prompts)
    current_texts = list(all_prompts)

    accumulated_ids = [[] for _ in range(total_num_prompts)]
    accumulated_lps = [[] for _ in range(total_num_prompts)]
    accumulated_mask = [[] for _ in range(total_num_prompts)] 
    accumulated_prm = [[] for _ in range(total_num_prompts)]
    finished = [False] * total_num_prompts

    for step in range(MAX_INTERACTION_STEPS):
        if all(finished): 
            break
        
        outputs = trainer.llm.generate(current_texts, sampling_params, use_tqdm=False)
        
        # if step == 0:
        #     print(f"\n[DEBUG] SamplingParams logprobs: {sampling_params.logprobs}", flush=True)
        #     first_res = outputs[0].outputs[0]
        #     print(f"[DEBUG] res.logprobs type: {type(first_res.logprobs)}", flush=True)
        #     if first_res.logprobs:
        #         print(f"[DEBUG] First token logprob data: {first_res.logprobs[0]}", flush=True)
        #     else:
        #         print(f"[DEBUG] res.logprobs is None or Empty!", flush=True)

        step_updates = [None] * total_num_prompts
        search_queries = []
        search_idx_map = []
        for i, output in enumerate(outputs):
            if finished[i]: 
                continue
            
            res = output.outputs[0]
            gen_text = res.text
            gen_ids = res.token_ids
            gen_lps = [lp[tid].logprob for tid, lp in zip(gen_ids, res.logprobs)] if res.logprobs else [0.0]*len(gen_ids)

            update_info = {
                "gen_text": gen_text,
                "gen_ids": gen_ids,
                "gen_lps": gen_lps,
                "ret_text": "", 
                "ret_ids": [],
                "is_finished": False
            }

            if "</answer>" in gen_text:
                update_info["is_finished"] = True
            elif "</subquery>" in gen_text:
                match = re.search(r"<subquery>(.*?)</subquery>", gen_text, re.DOTALL)
                if match:
                    query = match.group(1).strip()
                    search_queries.append(query)
                    search_idx_map.append(i)
            
            step_updates[i] = update_info

        if search_queries:
            search_results = trainer.retriever.batch_search(search_queries)
            for idx_in_batch, res_text in enumerate(search_results):
                original_idx = search_idx_map[idx_in_batch]
                formatted_ret = f"\n<retrieval>\n{res_text}\n</retrieval>"
                ret_ids = trainer.processing_class.encode(formatted_ret, add_special_tokens=False)
                
                step_updates[original_idx]["ret_text"] = formatted_ret
                step_updates[original_idx]["ret_ids"] = ret_ids
        
        for i in range(total_num_prompts):
            if finished[i] or step_updates[i] is None:
                continue
            
            info = step_updates[i]
            text_for_scoring = current_texts[i] + info["gen_text"] + info["ret_text"]
            step_score = trainer.get_prm_score(text_for_scoring) + format_reward_step(info["gen_text"]+info["ret_text"])
            
            accumulated_ids[i].extend(info["gen_ids"])
            accumulated_lps[i].extend(info["gen_lps"])
            accumulated_prm[i].extend([step_score] * len(info["gen_ids"]))
            accumulated_mask[i].extend([1] * len(info["gen_ids"])) 
            
            if info["ret_ids"]:
                accumulated_ids[i].extend(info["ret_ids"])
                accumulated_lps[i].extend([0.0] * len(info["ret_ids"]))
                accumulated_prm[i].extend([0.0] * len(info["ret_ids"]))
                accumulated_mask[i].extend([0] * len(info["ret_ids"])) 

            if info["is_finished"]:
                finished[i] = True
                finish_tag = "\n<|im_end|>"
                finish_ids = trainer.processing_class.encode(finish_tag, add_special_tokens=False)
                current_texts[i] = (text_for_scoring + finish_tag)
                accumulated_ids[i].extend(finish_ids)
                accumulated_lps[i].extend([0.0] * len(finish_ids))
                accumulated_prm[i].extend([0.0] * len(finish_ids))
                accumulated_mask[i].extend([0] * len(finish_ids))
            else:
                step_tag = "\n<step>\n"
                step_ids = trainer.processing_class.encode(step_tag, add_special_tokens=False)
                current_texts[i] = (text_for_scoring + step_tag)
                accumulated_ids[i].extend(step_ids)
                accumulated_lps[i].extend([0.0] * len(step_ids))
                accumulated_prm[i].extend([0.0] * len(step_ids))
                accumulated_mask[i].extend([0] * len(step_ids))

    if trainer.vllm_tensor_parallel_size > 1:
        local_rank = torch.distributed.get_rank(group=trainer.tp_group)
        start = local_rank * orig_size
        end = start + orig_size
        completion_ids = accumulated_ids[start:end]
        logprobs = accumulated_lps[start:end]
        prm_scores = accumulated_prm[start:end]
        completion_mask = accumulated_mask[start:end]
    else:
        completion_ids = accumulated_ids
        logprobs = accumulated_lps
        prm_scores = accumulated_prm
        completion_mask = accumulated_mask

    return {
        "prompt_ids": [trainer.processing_class.encode(p) for p in prompts], 
        "completion_ids": completion_ids, # List[List[int]]
        "logprobs": logprobs,             # List[List[float]]
        "rag_mask": completion_mask,      # List[List[int/float]]
        "prm_scores": prm_scores          # List[List[float]]
    }

def load_dataset_splits(test_size=100):
    data_list = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    
    full_dataset = Dataset.from_list(data_list)
    full_dataset = full_dataset.shuffle(seed=42) 
    dataset_dict = full_dataset.train_test_split(test_size=100, seed=42)   
    return dataset_dict['train'], dataset_dict['test']

if __name__ == "__main__":
    print("Loading data...")
    train_dataset, eval_dataset = load_dataset_splits()

    print("Loading Policy Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        dtype=torch.bfloat16, 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2"
    )

    # print("Loading PRM Model & Tokenizer...")
    # prm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side='left')
    # if prm_tokenizer.pad_token is None:
    #     prm_tokenizer.pad_token = prm_tokenizer.eos_token
    
    # prm_model = AutoModelForSequenceClassification.from_pretrained(
    #     PRM_PATH, 
    #     num_labels=1,
    #     dtype=torch.bfloat16,
    #     trust_remote_code=True
    # )
    # prm_model.eval()

    retriever = RemoteRetriever(RETRIEVAL_URL)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",     
        eval_steps=50,         
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        ddp_find_unused_parameters=False, 
        report_to="wandb",
        run_name="prorag",
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        log_completions=True,
        remove_unused_columns=False,
        use_vllm=True,
        vllm_mode="colocate"
    )

    print("Initializing Trainer...")
    trainer = RAGTrainer(
        model=model,
        args=config,
        reward_funcs=[outcome_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # reward_processing_classes=[prm_tokenizer],
        peft_config=lora_config,
        rollout_func=rag_rollout_with_prm,
        # reward_model=prm_model, 
        prm_beta=PRM_BETA,
        retriever=retriever
    )

    print("Start Training...")
    trainer.train()

    print("Training finished. Saving...")
    trainer.save_model(OUTPUT_DIR)