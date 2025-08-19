import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
import math
import collections
from typing import List

LORA_PATH = "/home/v-zhaowan/zhaowang/rag/save/final/final_adapter"
RAW_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_train.jsonl"
DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/raw/train_rl.json"
OUTPUT_PATH = "/home/v-zhaowan/zhaowang/rag/sample/data_mcts_1500.json"

DATA_START = 1000
DATA_LENGTH = 500
NUM_SIMULATIONS = 50
EXPANSION_WIDTH_K = 5
MAX_SEARCH_DEPTH = 6
C_PUCT = 2.5
LENGTH_PENALTY = 0.1

MAX_MODEL_INPUT_LENGTH = 2048
MAX_GENERATION_LENGTH = 512

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading SentenceTransformer model...")
E5_MODEL_NAME = 'intfloat/e5-large-v2'
similarity_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)

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

class E5VectorRetriever:
    def __init__(self, paragraphs: List[dict], model: SentenceTransformer):
        self.raw_paragraphs = paragraphs
        self.model = model
        
        self.corpus = [f'passage: {p.get("paragraph_text", "")}' for p in paragraphs]
        if not self.corpus:
            self.corpus_embeddings = None
            return

        self.corpus_embeddings = self.model.encode(
            self.corpus, 
            convert_to_tensor=True, 
            show_progress_bar=False
        ).to(DEVICE)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query:
            return []

        query_with_prefix = f'query: {query}'
        query_embedding = self.model.encode(query_with_prefix, convert_to_tensor=True).to(DEVICE)
        
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
        corpus_embeddings_norm = torch.nn.functional.normalize(self.corpus_embeddings, p=2, dim=1)
        
        cos_scores = torch.mm(query_embedding.unsqueeze(0), corpus_embeddings_norm.transpose(0, 1))[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus)))

        retrieved_docs_with_info = []
        for score, idx in zip(top_results[0], top_results[1]):
            paragraph = self.raw_paragraphs[idx]
            retrieved_docs_with_info.append(
                f"Document {paragraph['idx']} (Title: {paragraph['title']}): "
                f"{paragraph['paragraph_text']}"
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

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
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

        key_actions = {}
        for action in actions:
            answer_match = re.search(r"<answer>(.*?)</answer>", action, re.DOTALL)
            if answer_match:
                key = answer_match.group(1).strip()
            else:
                subquery_match = re.search(r"<subquery>(.*?)</subquery>", action, re.DOTALL)            
                key = subquery_match.group(1).strip() if subquery_match else action
            if key not in key_actions:
                key_actions[key] = action

        for key, action in key_actions.items():
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", action, re.DOTALL) 
            answer_match = re.search(r"<answer>(.*?)</answer>", action, re.DOTALL)
            if not subquery_match and answer_match:
                next_state = node.state + action
                prior_score = 0.5
                child = Node(state=next_state, parent=node, action=action, prior=prior_score, depth=node.depth + 1)
                node.children.append(child)
            else:
                prior_score = self._heuristic_function(key, node.state, self.question)
                next_state = node.state + action
                if subquery_match and action.strip().endswith("<retrieval>"):
                    retrieved_docs = self.retriever.retrieve(key, top_k=3)
                    retrieved_docs_text = "\n".join(retrieved_docs)
                    next_state += f"\n{retrieved_docs_text}\n</retrieval>\n"
                child = Node(state=next_state, parent=node, action=action, prior=prior_score, depth=node.depth + 1)
                node.children.append(child)

        return node.children

    def _simulate(self, node):
        current_state = node.state
        depth = node.depth
        while not self._is_terminal(current_state) and depth < MAX_SEARCH_DEPTH:
            response = generate(self.model, self.tokenizer, current_state, do_sample=False, max_gen_len=512)
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
    
    def _heuristic_function(self, subquery, prev_state, question):
        try:
            with torch.no_grad():
                subquery_embedding = similarity_model.encode(f'query: {subquery}', convert_to_tensor=True)
                question_embedding = similarity_model.encode(f'query: {question}', convert_to_tensor=True)
                score_rel = util.pytorch_cos_sim(subquery_embedding, question_embedding).item()

                prev_subquery_match = re.findall(r"<subquery>(.*?)</subquery>", prev_state, re.DOTALL)
                prev_subquery = prev_subquery_match[-1].strip() if prev_subquery_match else ""
                
                if prev_subquery:
                    prev_subquery_embedding = similarity_model.encode(f'query: {prev_subquery}', convert_to_tensor=True)
                    score_red = util.pytorch_cos_sim(subquery_embedding, prev_subquery_embedding).item()
                else:
                    score_red = 0.0

            retrieved_docs = self.retriever.retrieve(subquery, top_k=3)
            context = "\n".join(retrieved_docs)
            score_ans = get_nli_score(premise=context, hypothesis=subquery)
            
            final_score = score_ans * 0.5 + max(0, score_rel - score_red) * 0.5
            
            return final_score
        except Exception as e:
            print(f"Error in heuristic function: {e}")
            return 0.5

    def _is_terminal(self, state):
        return state.strip().endswith("<|im_end|>")

    def _compute_terminal_reward(self, state):
        answer_match = re.search(r"<answer>(.*?)</answer>", state, re.DOTALL)
        f1_score = 0.0

        if answer_match:
            extracted_answer = answer_match.group(1).strip()
            if extracted_answer and self.final_answer:
                scores = [calculate_f1_score(extracted_answer, ans) for ans in self.final_answer if ans]
                if scores:
                    f1_score = max(scores)

        if f1_score >= 0.9:
            return f1_score
        else:
            length = state.count("<step>")
            final_reward = f1_score - LENGTH_PENALTY * length
            if f1_score < 0.1 and length >= MAX_SEARCH_DEPTH:
                return -0.5
            return max(-1.0, final_reward)

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

if __name__ == "__main__":
    print("Loading generator model and tokenizer...")
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    custom_special_tokens = ["<step>", "</step>", "<subquery>", "</subquery>", "<retrieval>", "</retrieval>", "<subanswer>", "</subanswer>", "<answer>", "</answer>"]
    tokenizer.add_special_tokens({"additional_special_tokens": custom_special_tokens})
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    base_model.resize_token_embeddings(len(tokenizer))
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()

    print("Loading and preparing data...")
    raw_data = []
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            raw_data.append(item)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)[DATA_START: DATA_START+DATA_LENGTH]

    try:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
        print(f"Resuming from {len(new_data)} saved samples.")
    except:
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
        final_answer = [item.get("answer", "")]
        final_answer.extend(item.get("answer_aliases", []))
        if not final_answer:
            print(f"Warning: No answer for ID {idx}. Skipping.")
            continue
        
        init_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<step>\n"
        retriever = E5VectorRetriever(item["paragraphs"], similarity_model)

        mcts = MCTS(model, tokenizer, retriever, init_prompt, question, final_answer)
        mcts.run()
        search_tree = mcts.get_search_tree()
    
        new_data.append({
            "id": idx,
            "question": question,
            "mcts_tree": search_tree,
            "answer": final_answer
        })

        if len(new_data) % 10 == 0:
            print(f"\nSaving progress at {len(new_data)} samples...")
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

    print("Final saving...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)