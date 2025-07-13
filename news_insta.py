import requests
from bs4 import BeautifulSoup
import random
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import google.generativeai as genai
from diffusers import StableDiffusionPipeline
from PIL import Image
from instagrapi import Client

# ---------------------- CONFIG --------------------------
INSTAGRAM_USERNAME = "sigma_man_matrix"
INSTAGRAM_PASSWORD = "Brogoodsahil"
GENAI_API_KEY = "AIzaSyCJZaPICSqlw4Vx5tw4XgE1sZvLEo2-Vrg"
# --------------------------------------------------------

# Login to Instagram
cl = Client()
cl.login(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)
cl.dump_settings("insta_settings.json")

# Set up Gemini AI
genai.configure(api_key=GENAI_API_KEY)
for m in genai.list_models():
    print(m.name, "-", m.supported_generation_methods)

# Emoji set
emojis = ['📰', '📢', '🚨', '⚠️', '💥', '✈️', '🕊️', '🎯', '🔍', '😢']
valid_emojis = set(emojis)

# News data model
class NewsItem(BaseModel):
    title: str
    emoji: str
    source: str = "Times of India"
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("emoji")
    def validate_emoji(cls, v):
        if v not in valid_emojis:
            raise ValueError("Invalid emoji!")
        return v

# Scrape headlines
def get_headlines():
    URL = 'https://timesofindia.indiatimes.com/'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(URL, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.get_text(strip=True) for h in soup.select('h2') if len(h.get_text(strip=True)) > 15]
        return headlines[:10] if headlines else []
    except Exception as e:
        print(f"⚠️ Failed to scrape headlines: {e}")
        return []

headlines_raw = get_headlines() or [
    "India to launch AI policy by 2025",
    "New heatwave warning issued across North India",
    "ISRO plans mission to Venus by 2030",
    "Electric vehicle sales see 45% growth this year",
    "Government to boost semiconductor manufacturing",
    "Cyclone Biparjoy heads toward Gujarat coast",
    "NEET UG 2025 dates likely to change again",
    "Rain disrupts daily life in Mumbai",
    "Major tech layoffs shake Indian startup scene",
    "Apple opens new retail store in Delhi"
]

# Pick a headline + emoji
headline = random.choice(headlines_raw)
emoji = random.choice(emojis)
news_item = NewsItem(title=f"{headline} {emoji}", emoji=emoji)


model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Prompt engineering via Gemini
def ai_generate_prompt(headline: str) -> str:
    prompt = f"Generate a creative and imaginative image prompt for this news headline: '{headline}'"
    response = model.generate_content(prompt)
    return response.text.strip()


engineered_prompt = ai_generate_prompt(headline)
print(f"🎨 AI Prompt: {engineered_prompt}")

# Generate image via Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")
image = pipe(engineered_prompt).images[0]
image_path = "headline_image.png"
image.save(image_path)
print(f"✅ Image saved as '{image_path}'")

# Create caption
clean_caption = f"{headline} 📰 #AI #News #Trending #StableDiffusion #PythonAutomation"

# Post to Instagram
try:
    cl.photo_upload(image_path, clean_caption)
    print("✅ Posted to Instagram!")
except Exception as e:
    print(f"❌ Failed to post on Instagram: {e}")
