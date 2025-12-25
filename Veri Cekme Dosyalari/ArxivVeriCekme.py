import feedparser
import pandas as pd
from tqdm import tqdm
import time

TARGET = 3000
START = 0

rows = []

print("📚 Genel arXiv abstract çekiliyor...")

while len(rows) < TARGET:
    url = (
        "http://export.arxiv.org/api/query?"
        "search_query=all&"
        f"start={START}&max_results=100"
    )

    feed = feedparser.parse(url)

    if len(feed.entries) == 0:
        break

    for entry in feed.entries:
        abstract = entry.summary.replace("\n", " ").strip()
        wc = len(abstract.split())

        if 120 <= wc <= 180:
            # category bilgisi
            categories = [t.term for t in entry.tags]

            rows.append({
                "paper_id": entry.id,
                "title": entry.title,
                "abstract": abstract,
                "categories": ",".join(categories),
                "year": entry.published[:4],
                "label": "human"
            })

        if len(rows) >= TARGET:
            break

    START += 100
    time.sleep(3)

print("✅ Toplam çekilen abstract:", len(rows))

df = pd.DataFrame(rows)
df.to_csv("arxiv_human_3000.csv", index=False, encoding="utf-8")
print("📁 arxiv_human_3000.csv oluşturuldu")

def map_field(cats):
    if "cs." in cats:
        return "cs"
    if "math." in cats:
        return "math"
    if (
        "physics." in cats
        or "astro-ph" in cats
        or "hep-" in cats
        or "cond-mat" in cats
        or "quant-ph" in cats
        or "nucl-" in cats
    ):
        return "physics"
    return "other"

df["field"] = df["categories"].apply(map_field)
print(df["field"].value_counts())


df["field"] = df["categories"].apply(map_field)

print(df["field"].value_counts())

