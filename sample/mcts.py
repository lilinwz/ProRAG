import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np
import math
import collections
from typing import List

MODEL_PATH = "/home/aiscuser/ds/zhaowang/rag/save/sft/checkpoint-770"
DATA_PATH = "/home/v-zhaowan/ds/zhaowang/rag/data/2wiki/train.jsonl"
OUTPUT_PATH = "/home/v-zhaowan/zhaowang/rag/sample/tree.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_TREES = 5000
NUM_SIMULATIONS = 50
EXPANSION_WIDTH_K = 5
MAX_SEARCH_DEPTH = 20
C_PUCT = 2.5
LENGTH_PENALTY = 0.9

MAX_MODEL_INPUT_LENGTH = 2048
MAX_GENERATION_LENGTH = 1024

INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

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
    def __init__(self, tokenizer, stop_tokens: List[str], lookback_tokens: int=3):
        self.tokenizer = tokenizer
        self.stop_tokens = stop_tokens
        self.lookback_tokens = lookback_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] == 0:
            return False
        last_ids = input_ids[0, -self.lookback_tokens:].tolist()
        decoded = self.tokenizer.decode(last_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for tok in self.stop_tokens:
            if decoded.endswith(tok):
                return True
        return False

class E5VectorRetriever:
    def __init__(self, paragraphs: List[dict], model: SentenceTransformer):
        self.raw_paragraphs = paragraphs or []
        self.model = model
        self.corpus = [f'title: {p[0]}\npassage: {"\n".join(str(s) for s in p[1])}' for p in self.raw_paragraphs]
        if not self.corpus:
            self.corpus_embeddings = None
            return
        emb = self.model.encode(self.corpus, convert_to_tensor=False, show_progress_bar=False)
        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        self.corpus_embeddings = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True) + 1e-12
        self.corpus_embeddings = self.corpus_embeddings / norms

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query:
            return []
        q = f'query: {query}'
        q_emb = self.model.encode(q, convert_to_tensor=False, show_progress_bar=False)
        if hasattr(q_emb, "cpu"):
            q_emb = q_emb.cpu().numpy()
        q_emb = np.asarray(q_emb, dtype=np.float32)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        scores = np.dot(self.corpus_embeddings, q_emb)
        k = min(top_k, self.corpus_embeddings.shape[0])
        idxs = np.argsort(-scores)[:k]
        docs = []
        for idx in idxs:
            p = self.raw_paragraphs[int(idx)]
            docs.append(f"Document {p.get('idx','?')} (Title: {p.get('title','')}): {p.get('paragraph_text','')}")
        return docs

def generate(model, tokenizer, prompt, do_sample=True, max_gen_len=MAX_GENERATION_LENGTH):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    stop_tokens = ["<retrieval>", "</subanswer>", "<|im_end|>", "<|endoftext|>"]
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

    def backpropagate(self, init_reward):
        node = self
        reward = init_reward
        while node is not None:
            node.N += 1
            node.Q += reward
            reward = reward * LENGTH_PENALTY
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
            subanswer_match = re.search(r"<subanswer>(.*?)</subanswer>", action, re.DOTALL)
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", action, re.DOTALL)
            if answer_match:
                key = answer_match.group(1).strip()
            elif subquery_match:   
                key = subquery_match.group(1).strip()
            elif subanswer_match:      
                key = subanswer_match.group(1).strip()
            else:
                continue
            if key not in key_actions:
                key_actions[key] = action

        for key, action in key_actions.items():
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", action, re.DOTALL) 
            subanswer_match = re.search(r"<subanswer>(.*?)</subanswer>", action, re.DOTALL) 
            answer_match = re.search(r"<answer>(.*?)</answer>", action, re.DOTALL)
            if not subquery_match and subanswer_match and answer_match:
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
        em_score = 0.0

        if answer_match:
            extracted_answer = answer_match.group(1).strip()
            em_score = 1.0 if extracted_answer.strip().lower() == self.final_answer.strip().lower()

        return em_score

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
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    print("Loading and preparing data...")
    data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= NUM_TREES:
                break
            item = json.loads(line)
            data.append(item)

    try:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
        print(f"Resuming from {len(new_data)} saved samples.")
    except:
        new_data = []
        print("Starting a new sampling process.")
    
    processed_ids = {sample['id'] for sample in new_data}

    for sample in tqdm(data, desc="Sampling with MCTS"):
        idx = sample["_id"]
        if idx in processed_ids:
            continue

        question = sample["question"]
        answer = sample["answer"]
        
        user_content = INSTRUCTION_TEMPLATE.format(question=question)
        init_prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
        retriever = E5VectorRetriever(sample["context"], similarity_model)

        mcts = MCTS(model, tokenizer, retriever, init_prompt, question, answer)
        mcts.run()
        search_tree = mcts.get_search_tree()
    
        new_data.append({
            "id": idx,
            "question": question,
            "mcts_tree": search_tree,
            "answer": answer
        })

        if len(new_data) % 10 == 0:
            print(f"\nSaving progress at {len(new_data)} samples...")
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

    print("Final saving...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)