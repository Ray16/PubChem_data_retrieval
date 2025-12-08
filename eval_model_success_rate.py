import os
import json
from openai import OpenAI
from tqdm import tqdm
import pandas as pd 

ARGO_PORT = 8081

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url=f"http://127.0.0.1:{ARGO_PORT}/v1"
)

os.makedirs('answers', exist_ok=True)
os.makedirs('output', exist_ok=True) 

n_qa = 100 # number of questions to answer
n_answers = 5 # number of answers generated for each questions

columns_order = ["question", "answer_source", "correct_answer"] + [f'LLM_answer_{idx}' for idx in range(1,n_answers+1)]

df = pd.DataFrame(columns=columns_order)
output_path = 'output/LLM_answers.csv'

QA = json.load(open('chemistry_qa_dataset.json'))

for qa in tqdm(QA[:n_qa]):
    question = qa['question']
    correct_answer = qa['answer']
    answer_source = qa['answer_source']

    row_data = {
        "question": question,
        "answer_source": answer_source,
        "correct_answer": correct_answer
    }
    
    for i in range(n_answers):
        response = client.chat.completions.create(
            model="argo:gpt-4o-latest",
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {
                    "role": "system",
                    "content": 
                    "You are a scientific evaluation assistant. "
                    "You must return ONLY the final answer with NO explanation, "
                    "NO punctuation, and NO extra words."
                },
                {
                    "role": "user",
                    "content": 
                    f"{question}\n\n"
                    "Respond with exactly ONE of the following formats ONLY:\n"
                    "- A single integer (e.g., 3)\n"
                    "- A single decimal number (e.g., 12.5)\n"
                    "- A single word (e.g., Yes, No)\n"
                    "- A short chemical name (e.g., ether)\n\n"
                    "- If there are multiple items, connect them ONLY with the word and"
                    "Do NOT explain. Do NOT repeat the question."
                }
            ]
        )
        
        llm_answer = response.choices[0].message.content

        row_data[f"LLM_answer_{i+1}"] = llm_answer
        
    new_row_df = pd.DataFrame([row_data], columns=columns_order)
    
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    
print(f"\n✅ All Q&A pairs processed and final results saved to: {output_path}")