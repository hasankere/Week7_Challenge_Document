import sys
import os
import pandas as pd
# Append the project root path to sys.path
sys.path.append(os.path.abspath(".."))
# File path to the CSV
file_path = 'C:\\Users\\Hasan\\Desktop\\data science folder\\DoctorsET_data.csv'

# Load the CSV without headers
data = pd.read_csv(file_path)
print(data.head())
data['Message'].fillna("No message", inplace=True)
data['Media Path'].fillna("No media", inplace=True)

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.drop_duplicates(subset=['ID'], keep='first', inplace=True)
data['Channel Title'] = data['Channel Title'].str.strip().str.title()
data['Channel Username'] = data['Channel Username'].str.strip().str.lower()
data['Message'] = data['Message'].str.strip()
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Hour'] = data['Date'].dt.hour
import os

# Extract the filename from path
data['Media Filename'] = data['Media Path'].apply(lambda x: os.path.basename(x) if pd.notnull(x) else "No media")

# Check if files exist (optional)
data['Media Exists'] = data['Media Path'].apply(lambda x: os.path.exists(x) if pd.notnull(x) else False)
import matplotlib.pyplot as plt

data['Date'].dt.date.value_counts().sort_index().plot(kind='line', figsize=(10, 4), title="Messages Over Time")
plt.xlabel("Date")
plt.ylabel("Message Count")
plt.show()
from collections import Counter
import re

# Tokenize and count words
all_words = ' '.join(data['Message']).lower()
words = re.findall(r'\b\w+\b', all_words)
word_counts = Counter(words)

# Get the top 10 words
print(word_counts.most_common(10))
cleaned_file_path = 'C:\\Users\\Hasan\\Desktop\\data science folder\\DoctorsET_cleaned.csv'
data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")
from ultralytics import YOLO
import cv2
import os

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # "yolov8n.pt" is a lightweight pre-trained model
# Directory where images are stored
image_dir = "C:\\Users\\Hasan\\Desktop\\data science folder\\photos"
output_dir = "C:\\Users\\Hasan\\Desktop\\data science folder\\detected"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Filter dataset to only rows where media exists
df_images = data[data["Media Exists"] == True]

# Run YOLO on each image
for index, row in df_images.iterrows():
    image_path = os.path.join(image_dir, row["Media Filename"])
    
    # Check if image file exists
    if os.path.exists(image_path):
        print(f"Processing: {image_path}")
        
        # Run YOLO object detection
        results = model(image_path, save=True, save_dir=output_dir)
        
        # Save detected image path
        data.at[index, "Detected Image Path"] = os.path.join(output_dir, row["Media Filename"])
    else:
        print(f"File not found: {image_path}")

# Save the dataset with detected image paths
data.to_csv("C:\\Users\\Hasan\\Desktop\\DoctorsET_with_detections.csv", index=False)
print("Object detection complete!")
# Add an empty 'Detected Image Path' column if it's missing
if "Detected Image Path" not in data.columns:
    data["Detected Image Path"] = None
import os

output_dir = "C:\\Users\\Hasan\\Desktop\\data science folder\\photos"

# Check if the directory exists and contains images
if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
    print("✅ Detected images exist in:", output_dir)
else:
    print("❌ No detected images found! Re-run YOLO detection.")
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Test on a sample image from the web
results = model("https://ultralytics.com/images/zidane.jpg", show=True)

print("✅ YOLO model is working correctly!")
import pandas as pd

# Load the dataset
file_path = "C:\\Users\\Hasan\\Desktop\\DoctorsET_with_detections.csv"
data = pd.read_csv(file_path)

# Display columns to verify
print("✅ Data loaded successfully!")
print(data.columns)
import os

image_dir = "C:\\Users\\Hasan\\Desktop\\data science folder\\photos"

# Check if sample images exist
sample_files = data["Media Filename"].dropna().unique()[:5]  # First 5 images

for filename in sample_files:
    image_path = os.path.join(image_dir, filename)
    if os.path.exists(image_path):
        print(f"✅ File exists: {image_path}")
    else:
        print(f"❌ File missing: {image_path}")
import os
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define directories
image_dir = "C:\\Users\\Hasan\\Desktop\\data science folder\\photos"
output_dir = "C:\\Users\\Hasan\\Desktop\\data science folder\\detected"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Run YOLO on all images and save detected images
for index, row in data.iterrows():
    if row["Media Exists"]:  # Process only valid images
        image_path = os.path.join(image_dir, row["Media Filename"])

        if os.path.exists(image_path):
            print(f"Processing image: {image_path}")

            # Run YOLO object detection
            results = model(image_path)

            # Save detected image in the output directory
            detected_image_path = os.path.join(output_dir, row["Media Filename"])
            results.save(filename=detected_image_path)  # Save detected image

            # Update DataFrame
            data.at[index, "Detected Image Path"] = detected_image_path
        else:
            print(f"❌ Skipping missing file: {image_path}")

# Save the updated dataset
data.to_csv("C:\\Users\\Hasan\\Desktop\\DoctorsET_with_detections.csv", index=False)
print("✅ YOLO detection complete! CSV updated with detected image paths.")
import os
import pandas as pd
import nest_asyncio
import asyncio
from telethon import TelegramClient
import time

# Apply nest_asyncio to fix the event loop issue in Jupyter
nest_asyncio.apply()

# 🔑 Telegram API credentials (Replace with your own)
API_ID = "27904749"
API_HASH = "5d9f2efaa785a2c4d04b5043088c2813"
PHONE_NUMBER = "+251912026082"

# 🔗 Telegram channels to scrape
channels = [
    "https://t.me/DoctorsET",
    "https://t.me/lobelia4cosmetics",
    "https://t.me/yetenaweg",
    "https://t.me/EAHCI"
]

# 📂 Storage directory
output_dir = "C:\\Users\\Hasan\\Desktop\\data science folder"
os.makedirs(output_dir, exist_ok=True)

# 📌 Define the async function for scraping
async def scrape_telegram():
    for channel in channels:
        session_name = f"session_{channel.replace('https://t.me/', '')}"  # Unique session per channel
        async with TelegramClient(session_name, API_ID, API_HASH) as client:
            print(f"📥 Scraping data from {channel}...")

            # Get entity (channel ID)
            entity = await client.get_entity(channel)

            # Fetch messages
            messages = []
            async for message in client.iter_messages(entity, limit=500):  # Adjust limit as needed
                messages.append([
                    channel,
                    message.id,
                    message.date,
                    message.sender_id,
                    message.text,
                    message.photo.file_reference if message.photo else None
                ])

            # Save to CSV
            df = pd.DataFrame(messages, columns=["Channel", "Message_ID", "Date", "Sender_ID", "Text", "Media"])
            csv_path = os.path.join(output_dir, f"{channel.replace('https://t.me/', '')}.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ Data saved: {csv_path}")

    print("🚀 Scraping complete!")

# 🚀 Run the async function properly in Jupyter
asyncio.get_event_loop().run_until_complete(scrape_telegram())
import os
import pandas as pd
import nest_asyncio
import asyncio
from telethon import TelegramClient

# Apply nest_asyncio to fix the event loop issue in Jupyter
nest_asyncio.apply()

# 🔑 Telegram API credentials (Replace with your own)
API_ID = "27904749"
API_HASH = "5d9f2efaa785a2c4d04b5043088c2813"
PHONE_NUMBER = "+251912026082"

# 🔗 Telegram channels to scrape
channels = [
    "https://t.me/DoctorsET",
    "https://t.me/lobelia4cosmetics",
    "https://t.me/yetenaweg",
    "https://t.me/EAHCI"
]

# 📂 Storage directory
output_dir = "C:\\Users\\Hasan\\Desktop\\data science folder"
os.makedirs(output_dir, exist_ok=True)

# List to store all messages
all_messages = []

# 📌 Define the async function for scraping
async def scrape_telegram():
    for channel in channels:
        session_name = f"session_{channel.replace('https://t.me/', '')}"  # Unique session per channel
        async with TelegramClient(session_name, API_ID, API_HASH) as client:
            print(f"📥 Scraping data from {channel}...")

            # Get entity (channel ID)
            entity = await client.get_entity(channel)

            # Fetch messages
            async for message in client.iter_messages(entity, limit=500):  # Adjust limit as needed
                all_messages.append([
                    channel,
                    message.id,
                    message.date,
                    message.sender_id,
                    message.text,
                    message.photo.file_reference if message.photo else None
                ])

    # Save all collected messages to one CSV
    df = pd.DataFrame(all_messages, columns=["Channel", "Message_ID", "Date", "Sender_ID", "Text", "Media"])
    csv_path = os.path.join(output_dir, "all_channels_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ All data saved in: {csv_path}")

    print("🚀 Scraping complete!")

# 🚀 Run the async function properly in Jupyter
asyncio.get_event_loop().run_until_complete(scrape_telegram())
import pandas as pd

# Load data from CSV
df = pd.read_csv("C:\\Users\\Hasan\\Desktop\\data science folder\\all_channels_data.csv")

# Print actual column names to verify
print("Original Columns:", df.columns.tolist())

# Rename columns to match your expected format
df.rename(columns={
    "Channel Title": "channel",
    "Channel Username": "channel_username",
    "ID": "message.id",
    "Message": "message.text",
    "Date": "message.date",
    "Media Path": "message.photo.file_reference"
}, inplace=True)

# Convert 'message.date' to datetime format
df["message.date"] = pd.to_datetime(df["message.date"], errors="coerce")

# Display the first few rows to check the result
print(df.head())

# Optional: Save the cleaned DataFrame to a new CSV file
df.to_csv("cleaned_messages.csv", index=False)
import os
import logging
from telethon import TelegramClient
from telethon.tl.types import InputMessagesFilterPhotos
import asyncio

# Step 1: Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 2: Telegram API Credentials (Replace with your actual credentials)
API_ID = "27904749"  # Replace with your actual API ID
API_HASH = "5d9f2efaa785a2c4d04b5043088c2813"  # Replace with your actual API Hash
PHONE_NUMBER = "+251912026082"  # Replace with your phone number (for authentication)
SESSION_NAME = "telegram_scraper"

# Step 3: Define the Telegram channels to scrape images from
CHANNELS = [
    "Chemed_Telegram_Channel",  # Replace with the actual username of the channel
    "lobelia4cosmetics"
]

# Step 4: Temporary Directory to Store Raw Images
SAVE_DIR = "telegram_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Step 5: Initialize Telethon Client
client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

async def scrape_images():
    async with client:
        for channel in CHANNELS:
            logger.info(f"Fetching images from {channel}...")
            try:
                # Fetching the latest 100 messages from the channel with photos
                async for message in client.iter_messages(channel, filter=InputMessagesFilterPhotos, limit=100):
                    if message.photo:
                        # Download the photo and save it in the 'telegram_images' directory
                        file_path = os.path.join(SAVE_DIR, f"{message.photo.id}.jpg")
                        await message.download_media(file_path)
                        logger.info(f"Downloaded: {file_path}")
            except Exception as e:
                logger.error(f"Error while processing channel {channel}: {e}")
        
        logger.info("✅ Image scraping completed! Images are saved in the 'telegram_images' folder.")

# Step 6: Run the Async function to scrape images
asyncio.run(scrape_images())
import os
import pandas as pd

# Example of raw scraped data (replace with your actual data)
data = {
    "image_id": ["image_001", "image_002", "image_003", "image_002"],
    "file_name": ["img1.jpg", "img2.jpg", "img3.jpg", "img2.jpg"],
    "downloaded": [True, True, False, True],
    "channel": ["Chemed_Telegram_Channel", "lobelia4cosmetics", "Chemed_Telegram_Channel", "lobelia4cosmetics"]
}

# Create a DataFrame (you might want to load your data from a file or database)
df = pd.DataFrame(data)

# Remove duplicates based on 'image_id' and 'file_name'
df_cleaned = df.drop_duplicates(subset=["image_id", "file_name"])

# Handle missing values (e.g., fill missing 'downloaded' values with False)
df_cleaned["downloaded"].fillna(False, inplace=True)

# Standardize formats (e.g., convert all image file names to lowercase)
df_cleaned["file_name"] = df_cleaned["file_name"].str.lower()

# Validate data (ensure 'image_id' is unique)
assert df_cleaned["image_id"].is_unique, "Duplicate image IDs found!"

# Print the cleaned data
print(df_cleaned)
import sqlite3

# Create a database connection
conn = sqlite3.connect('telegram_data.db')
cursor = conn.cursor()

# Create a table for storing the cleaned data (if not exists)
cursor.execute("""
CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    file_name TEXT,
    downloaded BOOLEAN,
    channel TEXT
)
""")

# Insert the cleaned data into the database
df_cleaned.to_sql('images', conn, if_exists='replace', index=False)

# Commit and close the connection
conn.commit()
conn.close()
