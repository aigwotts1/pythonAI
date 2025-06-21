import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from json import dumps
import random
import pandas as pd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.cluster import KMeans

# Optional: Set emoji font for matplotlib if you have one installed
# plt.rcParams['font.family'] = 'Segoe UI Emoji'

# Emojis for news headlines
emojis = ['📰', '📢', '🚨', '⚠️', '💥', '✈️', '🕊️', '🎯', '🔍', '😢']
valid_emojis = set(emojis)

# Fetch a motivational quote from ZenQuotes API
quote_text = ""
try:
    quote_response = requests.get("https://zenquotes.io/api/random", timeout=10)
    quote_data = quote_response.json()[0]
    quote_text = f"“{quote_data['q']}” — {quote_data['a']}"
except Exception as e:
    quote_text = "“Start where you are. Use what you have. Do what you can.” — Arthur Ashe"
    print(f"[!] Failed to fetch quote: {e}")

# Pydantic model for a news item
class NewsItem(BaseModel):
    title: str
    emoji: str
    description: str = "No description available."
    source: str = "Hindustan Times"
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("emoji")
    def check_emoji(cls, v):
        if v not in valid_emojis:
            raise ValueError("Invalid emoji used!")
        return v

# Scrape top headlines from Hindustan Times
URL = 'https://www.hindustantimes.com'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(URL, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract top 10 headline texts
headlines_raw = []
for item in soup.select('h2'):
    text = item.get_text(strip=True)
    if len(text) > 15:
        headlines_raw.append(text)
    if len(headlines_raw) == 10:
        break

# Build structured headlines and validate with Pydantic
headlines = []
news_items = []

for i, headline in enumerate(headlines_raw, 1):
    emoji = random.choice(emojis)
    full_title = f"{i}) {headline} {emoji}"
    headlines.append(full_title)
    news_items.append(
        NewsItem(
            title=full_title,
            emoji=emoji
        )
    )

# Convert to DataFrame
df = pd.DataFrame([{
    "Title": item.title,
    "Emoji": item.emoji,
    "Timestamp": item.timestamp,
    "Title Length": len(item.title)
} for item in news_items])

# Shorten long titles for display
df_preview = df.copy()
df_preview["Title"] = df_preview["Title"].apply(lambda x: x[:60] + "..." if len(x) > 63 else x)

# NumPy stats
title_lengths = df["Title Length"].to_numpy()
length_mean = np.mean(title_lengths)
length_std = np.std(title_lengths)
length_max = np.max(title_lengths)
length_min = np.min(title_lengths)

# ✅ Matplotlib chart
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), title_lengths, color='skyblue')
plt.xlabel('Headline Index')
plt.ylabel('Title Length')
plt.title('Length of Each News Headline')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(range(1, 11))
plt.tight_layout()
plt.savefig('headline_lengths_chart.png')
plt.close()

# ✅ Seaborn plot with hue and legend=False
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=list(range(1, 11)), y=title_lengths, hue=list(range(1, 11)), palette="Blues_d", legend=False)
plt.xlabel('Headline Index')
plt.ylabel('Title Length')
plt.title('Seaborn: Length of Each News Headline')
plt.tight_layout()
plt.savefig('seaborn_headline_lengths.png')
plt.close()

# ✅ PyTorch Tensor operations on headline lengths
lengths_tensor = torch.tensor(title_lengths, dtype=torch.float32)
mean = lengths_tensor.mean()
std = lengths_tensor.std()
normalized_lengths = (lengths_tensor - mean) / std

pytorch_output = {
    "Original Lengths": title_lengths,
    "Tensor": lengths_tensor.tolist(),
    "Normalized": normalized_lengths.tolist(),
    "Mean": mean.item(),
    "Std Dev": std.item()
}

# ✅ Scikit-learn KMeans clustering (unsupervised)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[["Title Length"]])

# ✅ Cluster plot
plt.figure(figsize=(10, 6))
colors = sns.color_palette("Set2", 3)
for cluster in range(3):
    subset = df[df["Cluster"] == cluster]
    plt.scatter(subset.index, subset["Title Length"], color=colors[cluster], label=f"Cluster {cluster}")
plt.xlabel("Headline Index")
plt.ylabel("Title Length")
plt.title("KMeans Clustering on Headline Lengths")
plt.legend()
plt.tight_layout()
plt.savefig("headline_clusters.png")
plt.close()

# Write everything to file
with open("top_news_headlines.txt", "w", encoding="utf-8") as f:
    f.write("💡 Motivational Quote of the Day:\n")
    f.write(quote_text + "\n")
    f.write("-" * 40 + "\n\n")

    f.write("📰 Top 10 Headlines of the Day 🔥\n")
    f.write("-" * 40 + "\n\n")
    for line in headlines:
        f.write(line + "\n\n")

    f.write("-" * 40 + "\n")
    f.write("🧪 All 10 Validated NewsItems (Pydantic JSON):\n")
    f.write("[\n" + ",\n".join([item.model_dump_json(indent=2) for item in news_items]) + "\n]\n\n")

    f.write("📊 Sample News Data (Pandas DataFrame):\n")
    f.write(tabulate(df_preview, headers="keys", tablefmt="grid", showindex=False) + "\n\n")

    f.write("📈 Title Length Stats (via NumPy):\n")
    f.write(f"- Average Length: {length_mean:.2f} characters\n")
    f.write(f"- Standard Deviation: {length_std:.2f}\n")
    f.write(f"- Max Length: {length_max}\n")
    f.write(f"- Min Length: {length_min}\n\n")

    f.write("📉 Matplotlib Chart:\n")
    f.write("Headline length bar chart saved as 'headline_lengths_chart.png'\n")

    f.write("📊 Seaborn Chart:\n")
    f.write("Prettified headline length bar chart saved as 'seaborn_headline_lengths.png'\n")

    f.write("🧠 PyTorch Headline Length Tensor Info:\n")
    f.write(f"Original Lengths: {pytorch_output['Original Lengths']}\n")
    f.write(f"Normalized Tensor: {pytorch_output['Normalized']}\n")
    f.write(f"Mean: {pytorch_output['Mean']:.2f}, Std Dev: {pytorch_output['Std Dev']:.2f}\n\n")

    f.write("🔍 Headline Clustering (Scikit-learn KMeans):\n")
    f.write("Cluster results saved as 'headline_clusters.png'\n")
    f.write("Cluster assignments per headline:\n")
    for idx, row in df.iterrows():
        f.write(f"  - {row['Title']} => Cluster {row['Cluster']}\n")

print("✅ All data written to 'top_news_headlines.txt' and plots saved.")
