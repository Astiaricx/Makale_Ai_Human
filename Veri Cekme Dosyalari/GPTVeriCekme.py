import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ==============================
# 1) CLIENT
# ==============================
client = OpenAI()


# ==============================
# 2) PROMPTLAR
# ==============================
PROMPT_1 = """Rewrite the following scientific abstract in a formal academic style.
Do not use phrases like "this paper", "this study", or "we propose".
Preserve the original meaning but rephrase the text naturally.
Limit the output to 120–150 words.

Abstract:
"""

PROMPT_2 = """Paraphrase the following scientific abstract as if written by a graduate student.
Avoid common AI phrases such as "this paper" or "this work".
Keep the content precise, clear, and academically appropriate.
120–150 words.

Abstract:
"""

PROMPT_3 = """Rewrite the abstract below using varied sentence structures and natural academic language.
Do not explicitly mention the structure of the paper.
Avoid repetitive phrasing and AI-like expressions.
120–150 words.

Abstract:
"""

PROMPTS = {
    1: PROMPT_1,
    2: PROMPT_2,
    3: PROMPT_3
}

# ==============================
# 3) DATASET
# ==============================
df = pd.read_csv("human_fixed.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

parts = {
    1: df.iloc[:1000],
    2: df.iloc[1000:2000],
    3: df.iloc[2000:3000]
}

# ==============================
# 4) OUTPUT CSV (resume destekli)
# ==============================
OUTPUT_FILE = "ai_abstracts.csv"

if os.path.exists(OUTPUT_FILE):
    out_df = pd.read_csv(OUTPUT_FILE)
    done_ids = set(out_df["paper_id"])
else:
    out_df = pd.DataFrame()
    done_ids = set()

# ==============================
# 5) GPT CALL
# ==============================
def generate_ai_abstract(prompt, abstract):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # hızlı + yeterli kalite
        messages=[
            {"role": "system", "content": "You are an academic writing assistant."},
            {"role": "user", "content": prompt + abstract}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# ==============================
# 6) MAIN LOOP
# ==============================
for pid, part_df in parts.items():
    print(f"\n🚀 Prompt {pid} başlıyor...")

    for _, row in tqdm(part_df.iterrows(), total=len(part_df)):
        if row["paper_id"] in done_ids:
            continue

        try:
            ai_text = generate_ai_abstract(PROMPTS[pid], row["abstract"])

            new_row = {
                "paper_id": row["paper_id"],
                "human_abstract": row["abstract"],
                "ai_abstract": ai_text,
                "label": "ai",
                "prompt_id": pid
            }

            out_df = pd.concat([out_df, pd.DataFrame([new_row])], ignore_index=True)
            out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

            time.sleep(1.5)  # rate-limit güvenliği

        except Exception as e:
            print("⚠️ Hata:", e)
            time.sleep(10)

print("\n✅ AI abstract üretimi tamamlandı")
