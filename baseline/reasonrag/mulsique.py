import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from collections import Counter
from typing import List

MODEL_PATH = "reasonrag/Qwen2.5-7B-Instruct-ReasonRAG"
TEST_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_dev.jsonl"
E5_MODEL_NAME = 'intfloat/e5-large-v2'

MAX_MODEL_INPUT_LENGTH = 4096 
MAX_GENERATION_LENGTH = 512
MAX_HOP = 5 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


BEGIN_REASONING_PROMPT = """<|im_start|>system
You are an assistant for question answering with access to a retrieval tool. Upon receiving a question, your task is to:
* Analyze and Decompose the Question: Break the question into smaller, manageable sub-questions to ensure all aspects are addressed.
* Evaluate Your Knowledge: Assess each sub-question or component:
- Identify parts you can confidently answer based on your existing knowledge.
- Pinpoint parts that require additional information or verification through retrieval tools.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Respond Format:
If your knowledge is sufficient to answer the question, conclude with:
"So the answer is <answer>answer</answer>"
If retrieval is necessary to provide a complete answer, conclude with:
"So the next query is <query>query</query>"<|im_end|>
<|im_start|>user
Question: {question}<|im_end|>
<|im_start|>assistant
"""

DOCUMENT_ANALYSIS_PROMPT = """<|im_start|>system
You are an information retrieval assistant. Your task is to extract relevant evidence from the provided Wikipedia documents based on the latest query.

    Instructions:

    * Identify key terms or concepts in the query.
    * Search the documents for evidence that supports the query.
    * Response format:
    If relevant evidence is found, output:
       Based on the query, the relevant evidence is <evidence>evidence</evidence>.
    If no relevant evidence is found, output:
       <evidence>None</evidence>.<|im_end|>
<|im_start|>user
{history}

Reference: <reference>{reference}</reference><|im_end|>
<|im_start|>assistant
"""

REASONING_PROMPT = """<|im_start|>system
You are a question-answering assistant with access to a retrieval tool. Your goal is to provide a concise and accurate reasoning process.
Instructions:
* Error Reflection: If errors exist in previous thoughts, identify and correct them. Skip this step if no errors are present.
* Information Sufficiency: Evaluate whether the current information is sufficient to fully and accurately answer the question. If additional retrieval is needed, deconstruct the question and generate the next query. Avoid repeating previous queries. If no meaningful new query can be generated, explain why and provide an answer based on the current information.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Conclusion:
If generating an answer:
"So the answer is <answer>answer</answer>".
If more retrieval is needed:
"So the next query is <query>query</query>".<|im_end|>
<|im_start|>user
{history}<|im_end|>
<|im_start|>assistant
"""

ANSWER_GENERATION_PROMPT = """<|im_start|>system
You are a reasoning assistant with retrieval. Give a precise and very concise final answer for the given question, conclude with 'So the answer is <answer>answer</answer>'. Keep your final answer brief and to the point, followed without any explanation.<|im_end|>
<|im_start|>user
{history}<|im_end|>
<|im_start|>assistant
"""

class StopOnKeywords(StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens):
        self.tokenizer = tokenizer
        self.stop_token_ids = []
        for token in stop_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                self.stop_token_ids.extend(ids)
        self.stop_token_ids = list(set(self.stop_token_ids))

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
            self.corpus, convert_to_tensor=True, show_progress_bar=False
        ).to(DEVICE)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query: return []
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
                f"Document {paragraph['idx']} (Title: {paragraph['title']}): {paragraph['paragraph_text']}"
            )
        return retrieved_docs_with_info

def generate(model, tokenizer, prompt, max_input_len=MAX_MODEL_INPUT_LENGTH, max_gen_len=MAX_GENERATION_LENGTH):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    if input_ids.shape[1] > max_input_len:
        input_ids = input_ids[:, -max_input_len:]
    stop_tokens = ["<|im_end|>", "</query>", "</answer>", "</evidence>"]
    stopping_criteria = StoppingCriteriaList([StopOnKeywords(tokenizer, stop_tokens)])
    gen_output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_gen_len, 
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping_criteria
    )
    response = tokenizer.decode(gen_output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
    return response

def calculate_f1(prediction, ground_truth_answers):
    if not isinstance(ground_truth_answers, list): ground_truth_answers = [ground_truth_answers]
    max_f1 = 0.0
    prediction_tokens = prediction.lower().split()
    if not prediction_tokens: return 0.0
    for gt_ans in ground_truth_answers:
        ground_truth_tokens = gt_ans.lower().split()
        if not ground_truth_tokens: continue
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            current_f1 = 0.0
        else:
            precision = num_common / len(prediction_tokens)
            recall = num_common / len(ground_truth_tokens)
            current_f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, current_f1)
    return max_f1

def extract_tagged_content(tag: str, text: str) -> str | None:
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    if match: return match.group(1).strip()
    return None

def run_rag_inference(model, tokenizer, question, retriever, max_hops=MAX_HOP):
    thoughts = []
    full_history = f"Question: {question}"
    current_prompt = BEGIN_REASONING_PROMPT.format(question=question)
    final_answer = None

    for hop in range(max_hops):
        response = generate(model, tokenizer, current_prompt)
        thoughts.append(response)
        full_history += f"\n{response}"

        answer = extract_tagged_content("answer", response)
        if answer:
            final_answer = answer
            break
        
        query = extract_tagged_content("query", response)
        if query:
            retrieved_docs = retriever.retrieve(query, top_k=3)
            retrieved_text = "\n".join(retrieved_docs)
            current_prompt = DOCUMENT_ANALYSIS_PROMPT.format(history=full_history, reference=retrieved_text)
        else:
            current_prompt = REASONING_PROMPT.format(history=full_history)

    if final_answer is None:
        final_prompt = ANSWER_GENERATION_PROMPT.format(history=full_history)
        final_response = generate(model, tokenizer, final_prompt)
        thoughts.append(final_response)
        answer = extract_tagged_content("answer", final_response)
        final_answer = answer if answer else "Could not determine a final answer."

    return final_answer, thoughts

if __name__ == "__main__":
    print("Loading SentenceTransformer model...")
    similarity_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)

    print(f"Loading FULL model and tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"Loading test data from {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_samples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_samples)} test samples.")

    all_em_scores = []
    all_f1_scores = []
    for i, sample in tqdm(enumerate(test_samples), total=len(test_samples), desc="Running RAG inference"):
        question = sample["question"]
        golden_answers = [sample["answer"]] + sample["answer_aliases"]
        retriever = E5VectorRetriever(sample["paragraphs"], similarity_model)
        predicted_answer, full_generation_history = run_rag_inference(model, tokenizer, question, retriever, max_hops=MAX_HOP)
        
        em = 1.0 if predicted_answer in golden_answers else 0.0
        f1 = calculate_f1(predicted_answer, golden_answers)
        all_em_scores.append(em)
        all_f1_scores.append(f1)

        if i < 5:
            print(f"\n--- Sample {i+1} (ID: {sample['id']}) ---")
            print(f"Question: {question}")
            print(f"\nFull Generation History:")
            for hop_num, thought in enumerate(full_generation_history):
                print(f"--- HOP {hop_num+1} ---\n{thought.strip()}")
            print("\n" + "-"*15)
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Golden Answers: {golden_answers}")
            print(f"EM Score: {em:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 30)

    if all_em_scores:
        average_em = np.mean(all_em_scores)
        print(f"\n--- Overall RAG Performance ---")
        print(f"Total Samples Evaluated: {len(all_em_scores)}")
        print(f"Average EM Score: {average_em:.4f}")

    if all_f1_scores:
        average_f1 = np.mean(all_f1_scores)
        print(f"Average F1 Score: {average_f1:.4f}")
        print("-" * 30)