import json

def build_user_prompt(question):
    prompt = f"""You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""
    return prompt

def build_clean_system_prompt():
    prompt = """You are an expert question decomposition and reasoning analyst.
You will be given a text passage and must produce reasoning steps in natural language.

Your task:
Break down the content into several reasoning steps, each represented as a JSON object with the following structure:

{
    "new_chain_of_thought": [
        {
            "subquery": "... (a natural-language question about one factual aspect of the text; DO NOT include the answer or any hint of it)",
            "subanswer": "... (a concise factual answer)",
            "paragraph": "... (use the given text verbatim)"
        },
        ...
    ]
}

Guidelines:
- Each subquery must be a *complete sentence* question, clearly phrased, and must NOT contain the answer or suggest it.
- Each subanswer should be a concise factual response; it may be a single word, a short phrase, a number, or a full sentence — full sentences are allowed but not required.
- Each paragraph must directly use the original text verbatim, without any rewriting, paraphrasing, or summarization.
- Only include factual and inferable information from the text.
- Output **only** one valid JSON object following this schema, no extra text, explanations, or Markdown.
"""
    return prompt

def build_clean_user_prompt(query, chain, answer):
    prompt = f"""
Below are the fields. Produce ONE JSON object that matches the REQUIRED OUTPUT SCHEMA specified by the system message exactly.

query: {query}
chain_of_thought: {chain}
answer: {answer}

Remember:
- Rewrite each subquery into a complete question sentence.
- Provide subanswer as a concise factual response (may be a short phrase, number, or a full sentence).
- Provide paragraph as the original text (verbatim).
- Do not include any additional keys or text; the response must be exactly a single JSON object matching the schema.
"""
    return prompt

def build_cot_system_prompt():
    prompt = """You are a meticulous thought process simulator. Your task is to reconstruct the internal monologue of an AI agent solving a problem.

You will be given:
- The original question.
- A set of decomposed reasoning steps (`new_chain_of_thought`) containing the agent's actions (subqueries) and findings (paragraphs and subanswers).
- The final answer.

Your goal is to generate a step-by-step **internal monologue** that simulates how the agent would "think" its way from the question to the final answer.

Output a single JSON object in the following schema:

{
  "think_process": [
    {
      "id": "subquery_1",
      "think": "The initial thought process. Based on the user's question, what is the first logical thing I need to figure out and why?"
    },
    {
      "id": "subanswer_1",
      "think": "Simulate information extraction. Now that I have the search result, I will scan the paragraph and extract the key information that directly answers my first subquery."
    },
    {
      "id": "subquery_2",
      "think": "The next logical step. Given what I've just learned from the previous step, what is the immediate next question I must ask to move closer to the final answer?"
    },
    ...
    {
      "id": "final_answer",
      "think": "Synthesize the final answer. I now have all the pieces of information I need. I will combine my previous findings (subanswers) to construct the complete and final answer."
    }
  ]
}

**Crucial Guidelines:**

1.  **Adopt a First-Person Perspective:** Write each `think` step from the perspective of the AI agent (e.g., "Okay, first I need to understand...", "From this text, I can extract...", "Now that I know X, my next step is to find out Y..."). This simulates an active thought process.
2.  **Strictly Sequential Reasoning (No Future Peeking):** The thought process for any given step must *only* use information from the steps *before* it. You cannot justify asking `subquery_1` by mentioning what is in `subanswer_2` or the `final_answer`. Your reasoning must unfold sequentially, as if you don't know what's coming next.
3.  **Focus on "Thinking," Not "Explaining":**
    - For a **subquery**: Don't just explain *what* it is. Describe the reasoning that *leads* to it. (e.g., "The user is asking about A and B. I'll start by investigating A, as it seems to be the foundational element.")
    - For a **subanswer**: Describe the action of extracting the relevant fact from the provided paragraph. (e.g., "The paragraph discusses several dates, but the one relevant to my question is X, so I'll pull that out.")
    - For the **final answer**: Describe the process of synthesis. (e.g., "I've found piece A and piece B. Now I'll put them together to form the complete picture.")
4.  **Be Concise:** Each `think` entry should be 1–2 fluent sentences.
5.  **Output Format:** Output exactly one valid JSON object and nothing else.
"""
    return prompt

def build_cot_user_prompt(query, new_cot, answer):
    prompt = f"""Original question:
{query}

Decomposed reasoning steps (new_Chain_of_thought):
{json.dumps(new_cot, ensure_ascii=False, indent=2)}

Final answer:
{answer}

---

Your task is to reconstruct the internal monologue of an AI agent that solved this problem. Generate a step-by-step thinking process based on the information above.

Follow these crucial guidelines:
1.  **Adopt a First-Person Perspective:** Write each step from the agent's point of view (e.g., "Okay, first I need to find out...", "This paragraph tells me that...", "Now that I know X, my next logical step is to figure out Y...").
2.  **Strictly Sequential Reasoning (No Future Peeking):** Your thought process for any given step must *only* use information from the steps *before* it. Do not justify a step using knowledge from later steps or the final answer.
3.  **Focus on "Thinking," Not "Explaining":**
    - For a subquery: Describe the reasoning that *leads* to asking that question.
    - For a subanswer: Describe the action of *extracting* the relevant fact from the paragraph.
    - For the final answer: Describe the process of *synthesizing* your previous findings.

Output exactly one valid JSON object following the required schema and nothing else.
"""
    return prompt

def build_filter_system_prompt():
    prompt = """You are an expert AI Tutor creating a training dataset for a reasoning model.
Your task is to review the "History" and a list of "Candidate Next Steps" (generated by MCTS), and select exactly ONE pair of (Positive, Negative) actions.

### SELECTION RULES:
1. **Positive Action**: The most logical, efficient step that moves towards the answer.
2. **Negative Action**: A step that is clearly worse (e.g., hallucination, logic error, or **redundant** search for info already in history).
3. **Semantic Deduplication (CRITICAL)**: 
   - The Negative action MUST be semantically different from the Positive action.
   - Do NOT select a negative sample if it is just a rephrasing of the positive one (e.g. "In what year..." vs "In which year...").
4. **NO FORCED SELECTION**:
   - If all candidates are good, OR all are bad, OR the negatives are just duplicates of the positive:
   - **RETURN NULL** (set has_valid_pair to false). Do NOT force a selection.

### OUTPUT FORMAT (JSON):
{
    "has_valid_pair": true/false,
    "positive_id": <id of the best candidate>,
    "negative_id": <id of the bad candidate>,
    "reason": "<brief explanation>"
}
"""
    return prompt