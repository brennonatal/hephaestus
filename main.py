import getpass
import io
import logging
import os
import random
import time
import uuid

import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from PIL import Image

from prompts import GUIDE, IDEAS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Validate or prompt for the Groq API key
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = getpass.getpass("Enter your HuggingFace access token: ")


# Step 1: Select a random topic
topics = list(IDEAS.keys())
selected_topic = random.choice(topics)

# Step 2: Get the topic instructions
topic_instructions = IDEAS[selected_topic]

# Step 3: Set up the Groq LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Step 4: Create a ChatPromptTemplate with separate system and human messages
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are an image prompt generator specialized in FLUX models. Your task is to create detailed and effective image prompts based on the user's topic and instructions.  When generating the prompt, include all necessary details as per the instructions, and ensure that your output is ONLY the image prompt without any additional explanations or text!\n\n{GUIDE}",
        ),
        (
            "human",
            f"Please create a detailed image prompt for the following topic:\n\n{selected_topic}\n\n{topic_instructions}",
        ),
    ]
)

# Step 5: Chain the prompt with the LLM
chain = prompt | llm

# Step 6: Invoke the chain with the topic and instructions
logging.info(f"Requesting theme {selected_topic}...")
ai_msg = chain.invoke({"topic": selected_topic, "instructions": topic_instructions})

# Step 7: Print the output
image_prompt = ai_msg.content.strip()
logging.info(image_prompt)


API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": f"Bearer {os.environ["HF_TOKEN"]}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


logging.info("Generating image...")
generation_params = {
    "inputs": image_prompt,
    "parameters": {
        "num_inference_steps": 25,
        "guidance_scale": 3.5,
        "height": 1024,
        "width": 768,
    },
}

start_time = time.time()
image_bytes = query(generation_params)
logging.info(f"Image generated in {time.time() - start_time} seconds.")

# save image
image = Image.open(io.BytesIO(image_bytes))
topic_folder = selected_topic.replace(" ", "_").lower()
os.makedirs(topic_folder, exist_ok=True)
image.save(f"{topic_folder}/{uuid.uuid4()}.png")

logging.info(f"Image saved to {topic_folder}/")
