import os
import json
import re
import numpy as np
import math
import random
import asyncio
from typing import List, Dict
from prorag.utils.prompts import build_user_prompt
from prorag.utils.metric import calculate_f1_score
from prorag.utils.retriever import AsyncRemoteRetriever
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
import argparse

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
        
    def select_best_child(self, c_puct):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            score = (child.Q / (child.N + 1e-8)) + c_puct * child.prior * (math.sqrt(self.N) / (1 + child.N))
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def backpropagate(self, init_reward, gamma):
        node = self
        reward = init_reward
        while node is not None:
            node.N += 1
            node.Q += reward
            reward = reward * gamma
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
    def __init__(self, client: AsyncOpenAI, retriever: AsyncRemoteRetriever, initial_prompt, question, final_answer, args):
        self.client = client
        self.retriever = retriever
        self.root = Node(state=initial_prompt, depth=0)
        self.question = question
        self.final_answer = final_answer
        self.stop_tokens = ["</subquery>", "</subanswer>", "</answer>", "<|im_end|>"]
        self.num_simulations = args.num_simulations
        self.gamma = args.gamma
        self.c_puct = args.c_puct
        self.expansion_width = args.expansion_width
        self.max_depth = args.max_depth
        self.server = args.server
        self.max_completion_length = args.max_completion_length

    async def _async_retrieve(self, query):
        results = await self.retriever.batch_search([query])
        return results[0] if results else ""

    async def run(self):
        for _ in range(self.num_simulations):
            leaf_node = self._select(self.root)
            
            if self._is_terminal(leaf_node.action) or leaf_node.depth >= self.max_depth:
                reward = self._compute_terminal_reward(leaf_node.action)
                leaf_node.backpropagate(reward, self.gamma)
                continue

            child_nodes = await self._expand(leaf_node)
            
            if child_nodes:
                node_to_simulate = np.random.choice(child_nodes)
                reward = await self._simulate(node_to_simulate)
                node_to_simulate.backpropagate(reward, self.gamma)

    def _select(self, node):
        while node.is_fully_expanded() and not self._is_terminal(node.state) and node.depth < self.max_depth:
            node = node.select_best_child(self.c_puct)
        return node

    async def _expand(self, node):
        try:
            response = await self.client.completions.create(
                model=self.server,
                prompt=node.state,
                n=self.expansion_width,
                temperature=0.9,
                top_p=0.95,
                max_tokens=self.max_completion_length,
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
        
        while not self._is_terminal(current_action) and depth < self.max_depth:
            try:
                response = await self.client.completions.create(
                    model=self.server,
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

async def process_single_item(sem, client_pool, retriever, item, file_lock, args):
    client = client_pool[np.random.randint(0, len(client_pool))]
    
    async with sem:
        try:
            question = item["question"]
            answer = item["answer"]
            
            user_content = build_user_prompt(question)
            init_prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n" 
            
            mcts = AsyncMCTS(client, retriever, init_prompt, question, answer, args)
            await mcts.run()
            
            result = {
                "id": item["id"],
                "question": question,
                "mcts_tree": mcts.get_search_tree(),
                "answer": answer
            }

            async with file_lock:
                with open(args.output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            return result
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            return None

async def main(args):
    retriever = AsyncRemoteRetriever()

    data = []
    data_files = [p.strip() for p in args.data_path.split(",") if p.strip()]
    for path in data_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                data.append(item)
   
    random.shuffle(data)
    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                processed_ids.add(json.loads(line)['id'])
    
    data_to_process = [x for x in data if x['id'] not in processed_ids]
    data_to_process = data_to_process[:1000]
    print(f"Total: {len(data)}, Processed: {len(processed_ids)}, Remaining: {len(data_to_process)}")

    clients = [AsyncOpenAI(base_url=url, api_key=args.api_key) for url in args.api_urls]
    sem = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()
    
    tasks = []
    for item in data_to_process:
        tasks.append(process_single_item(sem, clients, retriever, item, file_lock, args))
    
    print(f"Starting execution with {args.concurrency} concurrent tasks...")
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks))
    for coro in pbar:
        await coro

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS / Tree Search Configuration")

    parser.add_argument("--server", type=str, default="rag-model", help="Model server name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data file(s)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output jsonl")

    parser.add_argument("--api_urls", nargs="+", default=[
        "http://localhost:8001/v1",
        "http://localhost:8002/v1",
        "http://localhost:8003/v1",
        "http://localhost:8004/v1"
    ], help="List of API endpoints")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API Key")

    parser.add_argument("--num_simulations", type=int, default=200, help="Number of MCTS simulations per step")
    parser.add_argument("--expansion_width", type=int, default=5, help="Max children to expand")
    parser.add_argument("--max_depth", type=int, default=10, help="Max search depth")
    parser.add_argument("--c_puct", type=float, default=2.5, help="Exploration constant c_puct")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--max_completion_length", type=int, default=1024, help="Max tokens for LLM generation")
    parser.add_argument("--concurrency", type=int, default=16, help="Async concurrency limit")

    args = parser.parse_args()
    asyncio.run(main(args))