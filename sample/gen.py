import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
import math

LORA_PATH = "/home/v-zhaowan/zhaowang/rag/save/final/final_adapter"
RAW_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_train.jsonl"
DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/raw/train_rl.json"
OUTPUT_PATH = "/home/v-zhaowan/zhaowang/rag/sample/sampled_data_mcts_real_nli.json"

NUM_SIMULATIONS = 200
EXPANSION_WIDTH_K = 5
MAX_SEARCH_DEPTH = 6
C_PUCT = 1.8
LENGTH_PENALTY = 0.1

MAX_MODEL_INPUT_LENGTH = 2048
MAX_GENERATION_LENGTH = 512

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading SentenceTransformer model...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

print("Loading NLI model for answerability scoring...")
NLI_MODEL_NAME = 'facebook/bart-large-mnli'
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(DEVICE)
nli_model.eval()

@torch.no_grad()
def get_nli_score(premise: str, hypothesis: str) -> float:
    if not premise or not hypothesis:
        return 0.1
    input_ids = nli_tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512).to(nli_model.device)
    logits = nli_model(input_ids).logits
    probs = logits.softmax(dim=1)
    return probs[:, 0].item()+probs[:, 2].item()

class StopOnKeywords(StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens):
        self.tokenizer = tokenizer
        self.stop_token_ids = []
        for token in stop_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                self.stop_token_ids.append(ids[0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] > 0 and input_ids[0, -1].item() in self.stop_token_ids:
            return True
        return False

class SimpleBM25Retriever:
    def __init__(self, paragraphs):
        self.raw_paragraphs = paragraphs
        self.corpus = [p["paragraph_text"] for p in paragraphs]
        if not self.corpus:
            self.bm25 = None
            return
        self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=3):
        if not self.bm25:
            return []
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        retrieved_docs_with_info = []
        for i in top_indices:
            retrieved_docs_with_info.append(
                f"Document {self.raw_paragraphs[i]['idx']} (Title: {self.raw_paragraphs[i]['title']}): "
                f"{self.raw_paragraphs[i]['paragraph_text']}"
            )
        return retrieved_docs_with_info

def generate(model, tokenizer, prompt, do_sample=True, max_gen_len=MAX_GENERATION_LENGTH):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    if input_ids.shape[1] > MAX_MODEL_INPUT_LENGTH:
        input_ids = input_ids[:, -MAX_MODEL_INPUT_LENGTH:]
    stop_tokens = ["<retrieval>", "<|im_end|>"]
    stopping_criteria = StoppingCriteriaList([StopOnKeywords(tokenizer, stop_tokens)])
    with torch.no_grad():
        gen_output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_gen_len, 
            do_sample=do_sample,
            temperature=0.7 if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
    response = tokenizer.decode(gen_output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
    return response

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

    def backpropagate(self, reward):
        node = self
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent


class MCTS:
    def __init__(self, model, tokenizer, retriever, initial_prompt, question, final_answer):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.root = Node(state=initial_prompt, depth=0)
        self.question = question
        self.final_answer = final_answer
        
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
        actions = []
        for _ in range(EXPANSION_WIDTH_K):
            generated_text = generate(self.model, self.tokenizer, node.state, do_sample=True)
            actions.append(generated_text)
        
        for action in set(actions):
            prior_score = self._heuristic_function(action, node.state, self.question)
            
            next_state = node.state + action
            if action.strip().endswith("<retrieval>"):
                subquery_match = re.search(r"<subquery>(.*?)</subquery>", action, re.DOTALL)
                if subquery_match:
                    subquery = subquery_match.group(1).strip()
                    retrieved_docs = self.retriever.retrieve(subquery, top_k=3)
                    retrieved_docs_text = "\n".join(retrieved_docs)
                    next_state += f"\n{retrieved_docs_text}\n</retrieval>\n"

            child = Node(state=next_state, parent=node, action=action, prior=prior_score, depth=node.depth + 1)
            node.children.append(child)
        return node.children

    def _simulate(self, node):
        current_state = node.state
        depth = node.depth
        while not self._is_terminal(current_state) and depth < MAX_SEARCH_DEPTH:
            response = generate(self.model, self.tokenizer, current_state, do_sample=False, max_gen_len=256)
            current_state += response
            
            if response.strip().endswith("<retrieval>"):
                subquery_match = re.search(r"<subquery>(.*?)</subquery>", response, re.DOTALL)
                if subquery_match:
                    subquery = subquery_match.group(1).strip()
                    retrieved_docs = self.retriever.retrieve(subquery, top_k=3)
                    retrieved_docs_text = "\n".join(retrieved_docs)
                    current_state += f"\n{retrieved_docs_text}\n</retrieval>\n"
            depth += 1
        
        return self._compute_terminal_reward(current_state)
    
    def _heuristic_function(self, action, prev_state, question):
        try:
            with torch.no_grad():
                action_embedding = similarity_model.encode(action, convert_to_tensor=True)
                question_embedding = similarity_model.encode(question, convert_to_tensor=True)
                score_rel = util.pytorch_cos_sim(action_embedding, question_embedding).item()

                prev_action_match = re.findall(r"<step>.*?</step>", prev_state, re.DOTALL)
                prev_action = prev_action_match[-1] if prev_action_match else ""
                
                if prev_action:
                    prev_action_embedding = similarity_model.encode(prev_action, convert_to_tensor=True)
                    score_red = util.pytorch_cos_sim(action_embedding, prev_action_embedding).item()
                else:
                    score_red = 0.0

            subquery_match = re.search(r"<subquery>(.*?)</subquery>", action, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                retrieved_docs = self.retriever.retrieve(subquery, top_k=1)
                context = "\n".join(retrieved_docs)
                # <<< MODIFIED: 调用真实的NLI评分函数 >>>
                score_ans = get_nli_score(premise=context, hypothesis=subquery)
            else:
                score_ans = 0.3
            
            w_g, w_p_rel, w_p_red = 1.2, 1.0, 0.8
            final_score = (w_g * score_ans) + (w_p_rel * score_rel) - (w_p_red * score_red)
            
            return 1 / (1 + math.exp(-final_score))
        except Exception as e:
            print(f"Error in heuristic function: {e}")
            return 0.5
    def _is_terminal(self, state):
        return "<|im_end|>" in state

    def _compute_terminal_reward(self, state):
        answer_match = re.search(r"<answer>(.*?)</answer>", state, re.DOTALL)
        if answer_match:
            extracted_answer = answer_match.group(1).strip().lower()
            if self.final_answer.lower() in extracted_answer or extracted_answer in self.final_answer.lower():
                is_correct = 1.0
            else:
                is_correct = 0.0
        else:
            is_correct = 0.0

        length = state.count("<step>")
        length_penalty = LENGTH_PENALTY * length
        
        if is_correct == 0.0 and length >= MAX_SEARCH_DEPTH:
            return -0.5
            
        return is_correct - length_penalty

    def get_best_samples(self, num_samples=3):
        paths = []
        def dfs(node):
            if self._is_terminal(node.state) or node.depth >= MAX_SEARCH_DEPTH:
                if node.N > 0:
                    paths.append({
                        "path": node.state,
                        "avg_q": node.Q / node.N,
                        "visits": node.N
                    })
                return

            for child in node.children:
                dfs(child)

        dfs(self.root)
        if not paths:
            return []
            
        sorted_paths = sorted(paths, key=lambda x: x['avg_q'], reverse=True)
        return [p['path'] for p in sorted_paths[:num_samples]]


# ==========================================================
# 4. 主执行逻辑
# ==========================================================

if __name__ == "__main__":
    print("Loading generator model and tokenizer...")
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    custom_special_tokens = ["<step>", "</step>", "<subquery>", "</subquery>", "<retrieval>", "</retrieval>", "<subanswer>", "</subanswer>", "<answer>", "</answer>"]
    tokenizer.add_special_tokens({"additional_special_tokens": custom_special_tokens})
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    print("Loading and preparing data...")
    raw_data = []
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            raw_data.append(item)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)[:2000]

    try:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
        print(f"Resuming from {len(new_data)} saved samples.")
    except (FileNotFoundError, json.JSONDecodeError):
        new_data = []
        print("Starting a new sampling process.")
    
    processed_ids = {sample['id'] for sample in new_data}

    for sample in tqdm(data, desc="Sampling with MCTS"):
        idx = sample["id"]
        if idx in processed_ids:
            continue
            
        item = raw_data[idx]
        if not item:
            print(f"Warning: ID {idx} not found in raw data. Skipping.")
            continue

        question = item["question"]
        final_answer = item.get("answer", "")
        if not final_answer:
            print(f"Warning: No answer for ID {idx}. Skipping.")
            continue
        
        init_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<step>\n"

        retriever = SimpleBM25Retriever(item["paragraphs"])

        mcts = MCTS(model, tokenizer, retriever, init_prompt, question, final_answer)
        mcts.run()
        sample_list = mcts.get_best_samples(num_samples=EXPANSION_WIDTH_K)
    
        new_data.append({
            "id": idx,
            "question": question,
            "samples": sample_list,
            "answer": final_answer
        })

        if len(new_data) % 10 == 0:
            print(f"\nSaving progress at {len(new_data)} samples...")
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

    print("Final saving...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)