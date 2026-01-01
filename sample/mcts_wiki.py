import os
import json
import re
import torch
import numpy as np
import math
import random
import collections
import asyncio
import aiohttp
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from functools import partial

MODEL_PATH = "/home/aiscuser/ds/zhaowang/rag/save/sft"
SERVED_MODEL_NAME = "rag-model" 

DATA1_PATH = "/home/aiscuser/ds/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_train.jsonl"
DATA2_PATH = "/home/aiscuser/ds/zhaowang/rag/data/HotpotQA/train.jsonl"
OUTPUT_PATH = "/home/aiscuser/ds/zhaowang/rag/data/raw/tree_new_wiki.jsonl"

API_BASE_URLS = [
    # "http://localhost:8000/v1",
    "http://localhost:8001/v1",
    "http://localhost:8002/v1",
    # "http://localhost:8003/v1"
]
API_KEY = "EMPTY"
RETRIEVAL_SERVER_URL = "http://localhost:8000/retrieve"

NUM_SIMULATIONS = 200
EXPANSION_WIDTH_K = 5
MAX_SEARCH_DEPTH = 10
C_PUCT = 2.5
Gamma = 0.99
MAX_GENERATION_LENGTH = 1024
MAX_CONCURRENT_TASKS = 16

RETRIEVER_DEVICE = "cuda:0" 
E5_MODEL_NAME = 'intfloat/e5-large-v2'

INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

class AsyncRemoteRetriever:
    def __init__(self, url: str, topk: int = 3):
        self.search_url = url
        self.topk = topk

    async def batch_search(self, queries: List[str]) -> List[str]:
        response = await self._batch_search(queries)
        results = response['result']
        return [self._passages2string(result) for result in results]

    async def _batch_search(self, queries):
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True 
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.search_url, json=payload, timeout=500) as response: 
                response.raise_for_status()
                return await response.json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

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
    def __init__(self, state, parent=None, action="", prior=0.0, depth=0):
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

    def to_dict(self):
        return {
            'action': self.action,
            'q': self.Q,
            'n': self.N,
            'prior': self.prior,
            'depth': self.depth,
            'children': [child.to_dict() for child in self.children]
        }

class AsyncMCTS:
    def __init__(self, client: AsyncOpenAI, retriever: AsyncRemoteRetriever, initial_prompt, question, final_answer):
        self.client = client
        self.retriever = retriever
        self.root = Node(state=initial_prompt, depth=0)
        self.question = question
        self.final_answer = final_answer
        self.stop_tokens = ["</subquery>", "</subanswer>", "</answer>", "<|im_end|>"]

    async def _async_retrieve(self, query):
        results = await self.retriever.batch_search([query])
        return results[0] if results else ""

    async def run(self, num_simulations=NUM_SIMULATIONS):
        for _ in range(num_simulations):
            leaf_node = self._select(self.root)
            
            if self._is_terminal(leaf_node.action) or leaf_node.depth >= MAX_SEARCH_DEPTH:
                reward = self._compute_terminal_reward(leaf_node.action)
                leaf_node.backpropagate(reward)
                continue

            child_nodes = await self._expand(leaf_node)
            
            if child_nodes:
                node_to_simulate = np.random.choice(child_nodes)
                reward = await self._simulate(node_to_simulate)
                node_to_simulate.backpropagate(reward)

    def _select(self, node):
        while node.is_fully_expanded() and not self._is_terminal(node.state) and node.depth < MAX_SEARCH_DEPTH:
            node = node.select_best_child()
        return node

    async def _expand(self, node):
        try:
            response = await self.client.completions.create(
                model=SERVED_MODEL_NAME,
                prompt=node.state,
                n=EXPANSION_WIDTH_K,
                temperature=0.9,
                top_p=0.95,
                max_tokens=MAX_GENERATION_LENGTH,
                stop=self.stop_tokens,
                extra_body={
                    "include_stop_str_in_output": True, 
                    "skip_special_tokens": False       
                }
            )
        except Exception as e:
            print(f"Expansion API Error: {e}")
            return []

        generated_texts = [choice.text for choice in response.choices]
        actions = list(dict.fromkeys(generated_texts))
        
        retrieval_tasks = []
        action_indices_needing_retrieval = []
        
        for idx, action in enumerate(actions):
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", action, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                retrieval_tasks.append(self._async_retrieve(subquery))
                action_indices_needing_retrieval.append(idx)

        retrieval_results = []
        if retrieval_tasks:
            retrieval_results = await asyncio.gather(*retrieval_tasks)

        retrieval_result_idx = 0    
        for idx, action in enumerate(actions):
            next_state = node.state + action
            new_action = "<step>\n" + action

            if idx in action_indices_needing_retrieval:
                retrieved_docs = retrieval_results[retrieval_result_idx]
                retrieval_result_idx += 1
                
                next_state += f"\n<retrieval>{retrieved_docs}\n</retrieval>"
                new_action += f"\n<retrieval>{retrieved_docs}\n</retrieval>"
           
            if not "</answer>" in action:
                next_state += "\n<step>\n"
                new_action += "\n"

            prior_score = 1.0 / len(actions)
            child = Node(state=next_state, parent=node, action=new_action, prior=prior_score, depth=node.depth + 1)
            node.children.append(child)
            
        return node.children

    async def _simulate(self, node):
        current_state = node.state
        current_action = node.action
        depth = node.depth
        
        while not self._is_terminal(current_action) and depth < MAX_SEARCH_DEPTH:
            try:
                response = await self.client.completions.create(
                    model=SERVED_MODEL_NAME,
                    prompt=current_state,
                    n=1,
                    temperature=0.0,
                    max_tokens=1024,
                    stop=self.stop_tokens,
                    extra_body={
                        "include_stop_str_in_output": True,
                        "skip_special_tokens": False
                    }
                )
            except Exception as e:
                print(f"Simulation API Error: {e}")
                break
                
            text = response.choices[0].text
            current_state += text
            current_action = "<step>\n" + text
            
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", text, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                retrieved_docs = await self._async_retrieve(subquery)
                current_state += f"\n<retrieval>{retrieved_docs}\n</retrieval>"
                current_action += f"\n<retrieval>{retrieved_docs}\n</retrieval>"
            
            if not "</answer>" in text:
                current_state += "\n<step>\n"
                current_action += "\n"

            depth += 1
        
        return self._compute_terminal_reward(current_action)

    def _is_terminal(self, state):
        return state.strip().endswith("<|im_end|>") or state.strip().endswith("</answer>")

    def _compute_terminal_reward(self, action):
        answer_match = re.search(r"<answer>(.*?)</answer>", action, re.DOTALL)
        f1 = 0.0
        if answer_match:
            extracted_answer = answer_match.group(1).strip()
            f1 = calculate_f1_score(extracted_answer, self.final_answer)
        return f1

    def get_search_tree(self) -> dict:
        return self.root.to_dict()

async def process_single_item(sem, client_pool, retriever, item, file_lock, output_file):
    client = client_pool[np.random.randint(0, len(client_pool))]
    
    async with sem:
        try:
            question = item["question"]
            answer = item["answer"]
            
            user_content = INSTRUCTION_TEMPLATE.format(question=question)
            init_prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n" 
            
            mcts = AsyncMCTS(client, retriever, init_prompt, question, answer)
            await mcts.run()
            
            result = {
                "id": item["id"],
                "question": question,
                "mcts_tree": mcts.get_search_tree(),
                "answer": answer
            }

            async with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            return result
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            return None

async def main():
    retriever = AsyncRemoteRetriever(RETRIEVAL_SERVER_URL)

    data = []
    with open(DATA1_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            if not item["id"].startswith("2hop"):
                data.append(item)
    
    with open(DATA2_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            if item.get("level") == "hard":
                data.append(item)
   
    random.shuffle(data)
    processed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except: pass
    
    data_to_process = [x for x in data if x['id'] not in processed_ids]
    data_to_process = data_to_process[:500]
    print(f"Total: {len(data)}, Processed: {len(processed_ids)}, Remaining: {len(data_to_process)}")

    clients = [AsyncOpenAI(base_url=url, api_key=API_KEY) for url in API_BASE_URLS]
    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    file_lock = asyncio.Lock()
    
    tasks = []
    for item in data_to_process:
        tasks.append(process_single_item(sem, clients, retriever, item, file_lock, OUTPUT_PATH))
    
    print(f"Starting execution with {MAX_CONCURRENT_TASKS} concurrent tasks...")
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks))
    for coro in pbar:
        await coro

if __name__ == "__main__":
    asyncio.run(main())