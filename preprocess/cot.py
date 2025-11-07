import json
import os
import asyncio
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI

# ===================== 路径配置 =====================
input_file = "/home/v-zhaowan/ds/zhaowang/rag/data/raw/mulsique_refined_v1.jsonl"
output_file = "/home/v-zhaowan/ds/zhaowang/rag/data/raw/mulsique_cot_v1.jsonl"

os.makedirs(os.path.dirname(output_file), exist_ok=True)
API_CACHE = {}

# ===================== Azure 配置 =====================
API_VERSION = "2024-10-21"
DEPLOYMENT_NAME = "gpt-4o_2024-11-20"
INSTANCE = "gcr/shared"
ENDPOINT = f"https://trapi.research.microsoft.com/{INSTANCE}"
SCOPE = "api://trapi/.default"

credential = get_bearer_token_provider(
    ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    ),
    SCOPE,
)

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    azure_ad_token_provider=credential,
    api_version=API_VERSION,
    timeout=60.0,
)

# ===================== Prompt =====================
SYSTEM_PROMPT_THINK = """
You are a meticulous thought process simulator. Your task is to reconstruct the internal monologue of an AI agent solving a problem.

You will be given:
- The original question.
- A set of decomposed reasoning steps (`new_CoT`) containing the agent's actions (subqueries) and findings (paragraphs and subanswers).
- The final answer.

Your goal is to generate a step-by-step **internal monologue** that simulates how the agent would "think" its way from the question to the final answer.

Output a single JSON object in the following schema:

{
  "think_process": [
    {
      "to": "subquery_1",
      "cot": "The initial thought process. Based on the user's question, what is the first logical thing I need to figure out and why?"
    },
    {
      "to": "subanswer_1",
      "cot": "Simulate information extraction. Now that I have the search result, I will scan the paragraph and extract the key information that directly answers my first subquery."
    },
    {
      "to": "subquery_2",
      "cot": "The next logical step. Given what I've just learned from the previous step, what is the immediate next question I must ask to move closer to the final answer?"
    },
    ...
    {
      "to": "final_answer",
      "cot": "Synthesize the final answer. I now have all the pieces of information I need. I will combine my previous findings (subanswers) to construct the complete and final answer."
    }
  ]
}

**Crucial Guidelines:**

1.  **Adopt a First-Person Perspective:** Write each `cot` step from the perspective of the AI agent (e.g., "Okay, first I need to understand...", "From this text, I can extract...", "Now that I know X, my next step is to find out Y..."). This simulates an active thought process.
2.  **Strictly Sequential Reasoning (No Future Peeking):** The thought process for any given step must *only* use information from the steps *before* it. You cannot justify asking `subquery_1` by mentioning what is in `subanswer_2` or the `final_answer`. Your reasoning must unfold sequentially, as if you don't know what's coming next.
3.  **Focus on "Thinking," Not "Explaining":**
    - For a **subquery**: Don't just explain *what* it is. Describe the reasoning that *leads* to it. (e.g., "The user is asking about A and B. I'll start by investigating A, as it seems to be the foundational element.")
    - For a **subanswer**: Describe the action of extracting the relevant fact from the provided paragraph. (e.g., "The paragraph discusses several dates, but the one relevant to my question is X, so I'll pull that out.")
    - For the **final answer**: Describe the process of synthesis. (e.g., "I've found piece A and piece B. Now I'll put them together to form the complete picture.")
4.  **Be Concise:** Each `cot` entry should be 1–2 fluent sentences.
5.  **Output Format:** Output exactly one valid JSON object and nothing else.
"""

def build_user_prompt_think_global(query, new_cot, answer):
    return f"""
Original question:
{query}

Decomposed reasoning steps (new_CoT):
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

# ===================== 异步调用 =====================
async def async_call_think_global(item, semaphore):
    uid = item.get("id")
    query = item.get("query")
    new_cot = item.get("new_chain_of_thought")
    answer = item.get("answer")

    if not query or not new_cot or not answer:
        print(f"❌ Skip {uid}: missing required fields")
        return None

    cache_key = f"{uid}_globalthink"
    if cache_key in API_CACHE:
        item["think_process"] = API_CACHE[cache_key]
        return item

    user_prompt = build_user_prompt_think_global(query, new_cot, answer)

    async with semaphore:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_THINK},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                top_p=0.9,
                max_tokens=4096,
            )
            raw_output = response.choices[0].message.content.strip()
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            parsed = json.loads(raw_output[start:end])

            think_chain = parsed.get("think_process", [])
            item["think_process"] = think_chain
            API_CACHE[cache_key] = think_chain
            print(f"[✔] Global think generated for {uid}")
            return item

        except Exception as e:
            print(f"[API Error in think global] {uid}: {e}")
            return None

# ===================== 主函数 =====================
async def main():
    # load input items
    with open(input_file, "r", encoding="utf-8") as fin:
        items = [json.loads(line) for line in fin]

    # load already processed ids from output file (if exists)
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as fout:
            for line in fout:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["id"])
                except:
                    continue

    # filter items to only those not yet processed
    items_to_process = [it for it in items if it["id"] not in processed_ids]
    print(f"✅ Total input: {len(items)}, to process: {len(items_to_process)}, already processed: {len(processed_ids)}")

    semaphore = asyncio.Semaphore(3)
    tasks = [asyncio.create_task(async_call_think_global(item, semaphore)) for item in items_to_process]

    # append new results to output_file
    with open(output_file, "a", encoding="utf-8") as fout:
        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    print("✅ Done! All global think processes saved to", output_file)

if __name__ == "__main__":
    asyncio.run(main())
