import requests
import os
import uuid
import time
import pyttsx3
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.config import change_settings

# --- AI Image Generation Imports ---
from diffusers import DiffusionPipeline
import torch
from PIL import Image
from PIL import ImageOps # Not directly used for this fix, but good general PIL import

# --- Configuration ---
NEWSDATA_IO_API_KEY = 'pub_4af990f083f04fd6a6d161dcdfb7d80d'
NEWS_API_URL = "https://newsdata.io/api/1/latest"
OUTPUT_VIDEO_DIR = "generated_reels"
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# --- ImageMagick Configuration ---
IMAGEMAGICK_BINARY_PATH = '/opt/homebrew/bin/magick' # Verify this path with 'which magick'
change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY_PATH})

# --- AI Image Generation Model Setup ---
ai_pipeline = None

try:
    print("Loading Stable Diffusion model... (This will download the model the first time)")
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS) for faster image generation!")
    else:
        device = "cpu"
        print("MPS (Apple Silicon GPU) not available. Using CPU (will be slower).")

    # --- CHANGE THIS LINE TO USE A DIFFERENT MODEL ---
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", # <-- CHANGED MODEL HERE!
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        safety_checker=None
    )

    pipeline.to(device)

    if device == "mps" and torch.backends.mps.is_available():
         pipeline.enable_attention_slicing()

    ai_pipeline = pipeline
    print("Stable Diffusion model loaded successfully.")

except Exception as e:
    print(f"Error loading Stable Diffusion model: {e}")
    print("AI Image generation will not work. Please ensure torch and diffusers are correctly installed.")
    print("For Apple Silicon, ensure PyTorch is installed with MPS support via the provided pip command.")
    ai_pipeline = None


# --- Step 1: Fetch News from API ---
def fetch_top_headline_from_api(api_key, api_url):
    print(f"Fetching headlines from {api_url} using API...")
    try:
        params = {
            'apikey': api_key,
            'language': 'en',
            'country': 'in'
        }
        response = requests.get(api_url, params=params)
        response.raise_for_status()

        data = response.json()

        if data and data.get('results'):
            top_article = data['results'][0]
            headline = top_article.get('title')
            if headline:
                print(f"Successfully fetched headline from API: {headline}")
                return headline
            else:
                print("First article found, but no title in its data.")
                return None
        else:
            print("No results found from the NewsData.io API.")
            print(f"API Response Status: {data.get('status')}")
            print(f"API Error Message: {data.get('message')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching from NewsData.io API: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing API response: {e}")
        return None

# --- Step 2: Headline Processing ---
def process_headline(headline, max_length_words=20):
    words = headline.split()
    if len(words) > max_length_words:
        return ' '.join(words[:max_length_words]) + '...'
    return headline

# --- Step 3: AI Text-to-Speech (Free & Offline) ---
def generate_audio_from_text_free(text, output_filename):
    print(f"Generating audio using pyttsx3 for: '{text}'...")
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    wav_filename = output_filename.replace(".mp3", ".wav")
    engine.save_to_file(text, wav_filename)
    engine.runAndWait()

    if os.path.exists(wav_filename) and os.path.getsize(wav_filename) > 0:
        print(f"Audio content written to file '{wav_filename}'")
        return wav_filename
    else:
        print(f"Failed to generate audio file: {wav_filename}")
        return None

# --- Step 4: AI Visual Generation (Manually resize PIL image with improved prompts) ---
def generate_video_visuals(headline_text, output_filename, duration=15):
    if ai_pipeline is None:
        print("Skipping AI image generation because the model failed to load.")
        clip = ColorClip((1920, 1080), color=(0,0,0), duration=duration)
        clip.write_videofile(output_filename, fps=24, codec="libx264")
        print(f"Fallback black screen video saved to {output_filename}")
        return output_filename

    print(f"Generating AI image for: '{headline_text}'...")
    try:
        positive_prompt = f"news broadcast studio, breaking news, {headline_text}, anchors desk, high quality, cinematic, realistic, detailed, vibrant colors, wide shot, professional photography, 4k"

        negative_prompt = "blurry, dark, gloomy, monochrome, ugly, distorted, low quality, grayscale, sketch, painting, poor lighting, pixelated, abstract, text, watermark, bad hands, deformed"

        generator = torch.Generator(device=ai_pipeline.device).manual_seed(42)

        pil_image = ai_pipeline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            generator=generator
        ).images[0]

        target_width, target_height = 1920, 1080
        img_width, img_height = pil_image.size
        target_aspect_ratio = target_width / target_height
        img_aspect_ratio = img_width / img_height

        if img_aspect_ratio > target_aspect_ratio:
            new_width = int(img_height * target_aspect_ratio)
            left = (img_width - new_width) / 2
            top = 0
            right = (img_width + new_width) / 2
            bottom = img_height
            cropped_image = pil_image.crop((left, top, right, bottom))
        elif img_aspect_ratio < target_aspect_ratio:
            new_height = int(img_width / target_aspect_ratio)
            left = 0
            top = (img_height - new_height) / 2
            right = img_width
            bottom = (img_height + new_height) / 2
            cropped_image = pil_image.crop((left, top, right, bottom))
        else:
            cropped_image = pil_image

        final_pil_image = cropped_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        image_path = os.path.join(OUTPUT_VIDEO_DIR, f"generated_image_{uuid.uuid4().hex[:8]}.png")
        final_pil_image.save(image_path)
        print(f"Generated and resized AI image saved to {image_path}")

        static_clip = ImageClip(image_path).set_duration(duration)

        static_clip.write_videofile(output_filename, fps=24, codec="libx264")
        print(f"Video from generated image saved to {output_filename}")
        return output_filename

    except Exception as e:
        print(f"Error during AI image generation: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to black screen video due to AI generation error.")
        clip = ColorClip((1920, 1080), color=(0,0,0), duration=duration)
        clip.write_videofile(output_filename, fps=24, codec="libx264")
        return output_filename


# --- Step 5: Video Assembly ---
def create_news_reel(visual_video_path, audio_path, headline_text, output_filename, duration=15):
    print(f"Assembling video '{output_filename}'...")
    try:
        video_clip = VideoFileClip(visual_video_path).subclip(0, duration)
        audio_clip = AudioFileClip(audio_path)

        if audio_clip.duration > duration:
            audio_clip = audio_clip.subclip(0, duration)
        elif audio_clip.duration < duration:
            pass

        video_clip = video_clip.set_audio(audio_clip)

        text_clip = TextClip(
            headline_text,
            fontsize=60,
            color='white',
            bg_color='rgba(0,0,0,0.6)',
            font='Arial-Bold',
            method='caption',
            size=(video_clip.w * 0.8, None),
            align='center'
        ).set_position(('center', video_clip.h * 0.75)).set_duration(duration).set_opacity(1)

        final_clip = CompositeVideoClip([video_clip, text_clip])

        final_clip.write_videofile(output_filename, fps=24, codec="libx264", audio_codec="aac")
        print(f"News reel created successfully at: {output_filename}")
        return output_filename

    except Exception as e:
        print(f"Error during video assembly: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Main Execution Flow ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    top_headline = fetch_top_headline_from_api(NEWSDATA_IO_API_KEY, NEWS_API_URL)

    if top_headline:
        print(f"\nOriginal Top Headline: {top_headline}")

        processed_headline = process_headline(top_headline, max_length_words=20)
        print(f"Processed Headline: {processed_headline}")

        unique_id = uuid.uuid4().hex[:8]
        audio_output_path = os.path.join(OUTPUT_VIDEO_DIR, f"headline_audio_{unique_id}.wav")
        visual_output_path = os.path.join(OUTPUT_VIDEO_DIR, f"ai_visuals_{unique_id}.mp4")
        final_reel_path = os.path.join(OUTPUT_VIDEO_DIR, f"news_reel_{unique_id}.mp4")

        audio_file = generate_audio_from_text_free(processed_headline, audio_output_path)

        visual_file = generate_video_visuals(processed_headline, visual_output_path)

        if audio_file and visual_file:
            created_video = create_news_reel(visual_file, audio_file, processed_headline, final_reel_path)
            if created_video:
                print(f"\nProject complete! Check out your reel: {created_video}")
            else:
                print("\nFailed to create the news reel.")
        else:
            print("\nCould not generate audio or visuals. Aborting video creation.")
    else:
        print("Failed to retrieve any headline.")