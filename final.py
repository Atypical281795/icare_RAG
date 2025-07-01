from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json

model_name = "Qwen/Qwen1.5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()


input_path = "001__.txt"
output_path = "001__qa_llm.json"


with open(input_path, "r", encoding="utf-8") as f:
    paragraphs = [p.strip() for p in re.split("[。！？]", f.read()) if len(p.strip()) > 20]

def extract_qa(paragraph):
    prompt = f"""你是一位醫療專家助手。請閱讀以下段落，整理並提出多組有意義的問答對。請使用「問:」與「答:」開頭，每一組問答請換行，內容避免口語與贅詞，每個答案請簡明扼要。

段落：
{paragraph}
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    qas = re.findall(r"問[:：](.+?)答[:：](.+?)(?=問[:：]|$)", result, re.DOTALL)
    return [(q.strip(), a.strip()) for q, a in qas if len(q.strip()) > 4 and len(a.strip()) > 4]

json_data = []
for para in paragraphs:
    qa_pairs = extract_qa(para)
    for q, a in qa_pairs:
        json_data.append({
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"共產出 {len(json_data)} 組問答，結果儲存為 JSON 格式於 {output_path}")