import os
import json
import re
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import math
import collections
from typing import List, Dict
import multiprocessing as mp
from vllm import LLM, SamplingParams

mp.set_start_method("spawn", force=True)

MODEL_PATH = "/home/aiscuser/ds/zhaowang/rag/save/sft/checkpoint-770"
DATA_PATH = "/home/aiscuser/ds/zhaowang/rag/data/HotpotQA/train.jsonl"
OUTPUT_PATH = "/home/aiscuser/ds/zhaowang/rag/data/raw/tree_hotpotqa.jsonl"

NUM_SIMULATIONS = 200
EXPANSION_WIDTH_K = 5
MAX_SEARCH_DEPTH = 10
C_PUCT = 2.5
Gamma = 0.99
MAX_GENERATION_LENGTH = 1024

NUM_GPUS = 4  
VLLM_GPU_MEMORY_UTILIZATION = 0.85

INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

class E5VectorRetriever:
    def __init__(self, paragraphs: Dict[str, List], model: SentenceTransformer):
        self.model = model
        self.corpus = []
        
        self.titles = paragraphs.get("title", [])
        self.passages = ['\n'.join(sentence) for sentence in paragraphs.get("sentences", [])]
        self.corpus = [f"Title: {t}\n{p}" for t, p in zip(self.titles, self.passages)]

        if not self.corpus:
            self.corpus_embeddings = None
            return

        emb = self.model.encode(self.corpus, convert_to_tensor=False, show_progress_bar=False)
        emb = np.asarray(emb)
        self.corpus_embeddings = emb.astype(np.float32)
        self.corpus_embeddings /= np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True) + 1e-12

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query:
            return []
        
        q_emb = self.model.encode(f"query: {query}", convert_to_tensor=False, show_progress_bar=False)
        q_emb = np.asarray(q_emb)
        q_emb = q_emb.astype(np.float32)
        q_emb /= np.linalg.norm(q_emb) + 1e-12

        scores = np.dot(self.corpus_embeddings, q_emb)
        k = min(top_k, len(scores))
        idxs = np.argsort(-scores)[:k]
        return [f"Title: {self.titles[idx]}\n{self.passages[idx]}" for idx in idxs]

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
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

class Node:
    def __init__(self, state, parent=None, action=None, prior=0.0, depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.depth = depth
        self.Q = 0.0
        self.N = 0
        self.prior = prior

    def is_fully_expanded(self):
        return len(self.children) > 0
        
    def select_best_child(self, C_puct=C_PUCT):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            score = (child.Q / (child.N + 1e-8)) + C_puct * child.prior * (math.sqrt(self.N) / (1 + child.N))
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def backpropagate(self, init_reward):
        node = self
        reward = init_reward
        while node is not None:
            node.N += 1
            node.Q += reward
            reward = reward * Gamma
            node = node.parent

class MCTS:
    def __init__(self, llm_engine, retriever, initial_prompt, question, final_answer):
        self.llm = llm_engine
        self.retriever = retriever
        self.root = Node(state=initial_prompt, depth=0)
        self.question = question
        self.final_answer = final_answer
        
        self.stop_tokens = ["<retrieval>", "</subanswer>", "<|im_end|>", "<|endoftext|>"]

    def run(self, num_simulations=NUM_SIMULATIONS):
        for _ in range(num_simulations):
            leaf_node = self._select(self.root)
            
            if self._is_terminal(leaf_node.state) or leaf_node.depth >= MAX_SEARCH_DEPTH:
                reward = self._compute_terminal_reward(leaf_node.state)
                leaf_node.backpropagate(reward)
                continue

            child_nodes = self._expand(leaf_node)
            
            if child_nodes:
                node_to_simulate = np.random.choice(child_nodes)
                reward = self._simulate(node_to_simulate)
                node_to_simulate.backpropagate(reward)

    def _select(self, node):
        while node.is_fully_expanded() and not self._is_terminal(node.state) and node.depth < MAX_SEARCH_DEPTH:
            node = node.select_best_child()
        return node

    def _expand(self, node):
        sampling_params = SamplingParams(
            n=EXPANSION_WIDTH_K,
            temperature=0.9,
            top_p=0.95,
            max_tokens=MAX_GENERATION_LENGTH,
            stop=self.stop_tokens,
            skip_special_tokens=False,
            include_stop_str_in_output=True
        )
        
        outputs = self.llm.generate([node.state], sampling_params, use_tqdm=False)
        generated_texts = [output.text for output in outputs[0].outputs]

        actions = list(dict.fromkeys(generated_texts))
        
        for action in actions:
            next_state = node.state + action

            subquery_match = re.search(r"<subquery>\n(.*?)</subquery>", action, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                retrieved_docs = self.retriever.retrieve(subquery, top_k=3)
                retrieved_docs_text = "\n".join(retrieved_docs)
                next_state += f"{retrieved_docs_text}\n</retrieval>\n"
            
            prior_score = 1.0 / len(actions)
            child = Node(state=next_state, parent=node, action=action, prior=prior_score, depth=node.depth + 1)
            node.children.append(child)
            
        return node.children

    def _simulate(self, node):
        current_state = node.state
        depth = node.depth
        
        while not self._is_terminal(current_state) and depth < MAX_SEARCH_DEPTH:
            sampling_params = SamplingParams(
                n=1,
                temperature=0.0,
                max_tokens=1024,
                stop=self.stop_tokens,
                skip_special_tokens=False,
                include_stop_str_in_output=True
            )
            
            outputs = self.llm.generate([current_state], sampling_params, use_tqdm=False)
            response = outputs[0].outputs[0].text
            current_state += response
            
            if re.search(r"<subquery>\n(.*?)</subquery>", response, re.DOTALL):
                subquery_match = re.search(r"<subquery>\n(.*?)</subquery>", response, re.DOTALL)
                if subquery_match:
                    subquery = subquery_match.group(1).strip()
                    retrieved_docs = self.retriever.retrieve(subquery, top_k=3)
                    retrieved_docs_text = "\n".join(retrieved_docs)
                    current_state += f"{retrieved_docs_text}\n</retrieval>\n"
            
            depth += 1
        
        return self._compute_terminal_reward(current_state)

    def _is_terminal(self, state):
        return state.strip().endswith("<|im_end|>")

    def _compute_terminal_reward(self, state):
        answer_match = re.search(r"<answer>\n(.*?)</answer>", state, re.DOTALL)
        f1 = 0.0
        if answer_match:
            extracted_answer = answer_match.group(1).strip()
            f1 = calculate_f1_score(extracted_answer, self.final_answer)
        return f1

    def _node_to_dict(self, node: Node) -> dict:
        if node is None:
            return None
        node_representation = {
            'action': node.action,
            'q': node.Q,
            'n': node.N,
            'prior': node.prior,
            'depth': node.depth,
            'children': [self._node_to_dict(child) for child in node.children]
        }
        return node_representation

    def get_search_tree(self) -> dict:
        return self._node_to_dict(self.root)

def worker_process(gpu_id, process_idx, data_slice, part_file_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Proc {process_idx} (GPU {gpu_id}): Loading Models...")
    
    E5_MODEL_NAME = 'intfloat/e5-large-v2'
    similarity_model = SentenceTransformer(E5_MODEL_NAME, device='cuda:0') 

    llm = LLM(
        model=MODEL_PATH, 
        tokenizer=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1, 
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        enforce_eager=False
    )
    
    processed_ids_in_part = set()
    if os.path.exists(part_file_path):
        with open(part_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed_ids_in_part.add(item['id'])
                except:
                    continue
    print(f"Proc {process_idx}: Found {len(processed_ids_in_part)} items already in {part_file_path}")

    for sample in tqdm(data_slice, desc=f"Proc {process_idx}"):
        try:
            idx = sample["id"]
            
            if idx in processed_ids_in_part:
                continue

            question = sample["question"]
            answer = sample["answer"]
            
            user_content = INSTRUCTION_TEMPLATE.format(question=question)
            init_prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n" 
            retriever = E5VectorRetriever(sample["context"], similarity_model)

            mcts = MCTS(llm, retriever, init_prompt, question, answer)
            mcts.run()
            search_tree = mcts.get_search_tree()
            
            result_item = {
                "id": idx,
                "question": question,
                "mcts_tree": search_tree,
                "answer": answer
            }
            
            with open(part_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            continue

if __name__ == "__main__":
    data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: 
                continue
            item = json.loads(line)
            if item.get("level") == "hard":
                data.append(item)

    processed_ids = set()
    try:
        existing_data = []
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing_data.append(json.loads(line))
        print(f"Resuming from {len(existing_data)} saved samples in main file.")
        for item in existing_data:
            processed_ids.add(item['id'])
    except:
        existing_data = []
        
    temp_files = [f"{OUTPUT_PATH}.part{i}" for i in range(NUM_GPUS)]
    for tf in temp_files:
        if os.path.exists(tf):
            with open(tf, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        processed_ids.add(item['id'])
                    except:
                        pass
    print(f"Total processed samples found (Main + Parts): {len(processed_ids)}")

    data = [x for x in data if x["id"] not in processed_ids]
    num_processes = NUM_GPUS 
    chunk_size = math.ceil(len(data) / num_processes)
    processes = []

    print(f"Starting {num_processes} processes (1 per GPU)...")

    for i in range(num_processes):
        gpu_id = i % NUM_GPUS
        chunk = data[i*chunk_size : (i+1)*chunk_size]
        part_file_path = temp_files[i]
        
        if not chunk:
            continue
        
        p = mp.Process(target=worker_process, args=(gpu_id, i, chunk, part_file_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes finished. Merging files...")
    
    final_data = existing_data
    
    for tf in temp_files:
        if os.path.exists(tf):
            with open(tf, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        final_data.append(json.loads(line))
                    except:
                        continue

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(final_data)} samples.")