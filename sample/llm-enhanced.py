import json
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")  # 换成你的key

# 原始正样本
sample = { 
    "id": 710,
    "user": "Where did the chief supporter of the bill that got Jefferson motivated to draft his Statute deliver his memorable speech?",
    "assistant": [
        "<step>\n",
        "The original question asks where the chief supporter of the bill that motivated Jefferson to draft his Statute delivered his memorable speech...",
        "</answer>\n",
        "St. John's Church",
        "</answer>\n"
    ]
}

# 负样本生成提示
negative_prompt = f"""
You are given a QA pair. The QA is correct and well-reasoned.
I need you to generate **three different negative samples** by altering the reasoning or the answer so that the final answer is wrong.

Rules:
1. Keep the question format the same or very similar.
2. Modify intermediate reasoning steps or retrieval content so that the conclusion is wrong.
3. Maintain plausible-looking reasoning — the wrong answer should appear believable.
4. Do not simply output 'wrong' or nonsense; keep it historically plausible but incorrect.

Original QA:
Question: {sample['user']}
Answer trace:
{''.join(sample['assistant'])}

Output format:
[
  {{
    "id": "710_neg1",
    "user": "...",
    "assistant": ["<step> ... </step>", "...", "<answer>Wrong Answer</answer>"]
  }},
  ...
]
"""

resp = client.chat.completions.create(
    model="gpt-4o-mini",  # 换你可用的模型
    messages=[
        {"role": "system", "content": "You are a data augmentation assistant for QA datasets."},
        {"role": "user", "content": negative_prompt}
    ],
    temperature=0.9
)

# 解析模型输出
negatives = json.loads(resp.choices[0].message.content)

# 保存到文件
with open("negative_samples.json", "w", encoding="utf-8") as f:
    json.dump(negatives, f, indent=2, ensure_ascii=False)

print("Generated negative samples saved to negative_samples.json")
