import json

input_path = "/home/v-zhaowan/ds/zhaowang/rag/data/raw/mulsique_cot_v1.jsonl"
output_path = "/home/v-zhaowan/ds/zhaowang/rag/data/train_sft.jsonl"

with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in fin:
        item = json.loads(line)
        
        idx = item["id"]
        question = item["query"]
        thought = item["think_process"]
        response = []
        try:
            for i, subitem in enumerate(item["new_chain_of_thought"]):
                response.append(f"<step>\n{thought[2*i]['cot']}</step>\n<subquery>\n{subitem['subquery']}</subquery>\n")
                response.append(f"<retrieval>\n{subitem['paragraph']}</retrieval>\n")
                response.append(f"<step>\n{thought[2*i+1]['cot']}</step>\n<subanswer>\n{subitem['sub-answer']}</subanswer>\n")
            
            if thought[-1]['to'] == "final_answer":
                response.append(f"<step>\n{thought[-1]['cot']}</step>\n<answer>\n{item['answer']}</answer>\n") 
            else:
                continue
            out = {
                "id": idx,
                "question": question,
                "response": "".join(response).strip()
            }   
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        except:
            continue
