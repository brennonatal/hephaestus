import getpass
import io
import json
import logging
import os
import random
import sys
import tempfile
import time
import uuid

import requests
from gradio_client import Client, handle_file
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from PIL import Image
from pydantic import BaseModel, Field
from retry import retry

from prompts import GUIDE, IDEAS


# Define the Pydantic model
class ImagePrompt(BaseModel):
    """Detailed image prompt for FLUX model."""

    final_prompt: str = Field(description="The final detailed image prompt.")


def main():
    """Main function to generate an image based on a randomly selected topic."""
    setup_logging()
    validate_api_keys()

    # Step 1: Select a random topic
    selected_topic = select_random_topic(IDEAS)

    # Step 2: Get the topic instructions
    topic_instructions = IDEAS[selected_topic]

    # Step 3: Set up the Groq LLM
    llm = setup_groq_llm()

    # Wrap the LLM with structured output
    structured_llm = llm.with_structured_output(ImagePrompt)

    # Get the JSON schema
    schema = ImagePrompt.model_json_schema()

    # Step 4: Create a ChatPromptTemplate with separate system and human messages
    prompt = create_chat_prompt_template()

    # Step 5: Chain the prompt with the structured LLM
    chain = prompt | structured_llm

    # Step 6: Invoke the chain with the topic and instructions
    image_prompt_data = generate_image_prompt(
        chain, GUIDE, selected_topic, topic_instructions, schema
    )

    # Step 7: Extract the final prompt from the validated data
    image_prompt = image_prompt_data.final_prompt.strip()
    logging.info(f"Generated image prompt:\n{image_prompt}")

    # Step 8: Generate the image
    image_bytes = generate_image(image_prompt)

    # Step 9: Upscale the image
    upscaled_image_bytes = upscale_image(image_bytes)

    # Step 10: Save the upscaled image
    save_image(selected_topic, upscaled_image_bytes)


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def validate_api_keys():
    """Ensure that the necessary API keys are set, prompting the user if not."""
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
    if "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = getpass.getpass(
            "Enter your HuggingFace access token: "
        )


def select_random_topic(ideas_dict):
    """Select a random topic from the IDEAS dictionary."""
    topics = list(ideas_dict.keys())
    selected = random.choice(topics)
    logging.info(f"Selected topic: {selected}")
    return selected


def setup_groq_llm():
    """Initialize the Groq LLM with specified parameters."""
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def create_chat_prompt_template():
    """Create a ChatPromptTemplate with system and human messages."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an image prompt generator specialized in FLUX models. "
                    "Your task is to create detailed and effective image prompts based on the user's topic and instructions. "
                    "Output the final prompt in JSON format matching the following schema:\n\n"
                    "{schema}\n\n"
                    "Ensure that your output is ONLY the JSON object without any additional explanations or text!\n\n{GUIDE}"
                ),
            ),
            (
                "human",
                (
                    "Please create a detailed image prompt for the following topic:\n\n"
                    "{topic}\n\n{instructions}"
                ),
            ),
        ]
    )


@retry(Exception, delay=1, backoff=2, tries=3)
def generate_image_prompt(chain, guide, topic, instructions, schema):
    """Generate an image prompt using the structured LLM chain."""
    logging.info(f"Requesting image prompt for topic: '{topic}'...")
    try:
        image_prompt_data = chain.invoke(
            {
                "GUIDE": guide,
                "schema": json.dumps(schema),
                "topic": topic,
                "instructions": instructions,
            }
        )
    except Exception as e:
        logging.error(f"Error generating image prompt: {e}")
        sys.exit(1)

    return image_prompt_data


def save_image(selected_topic, image_bytes):
    """Save the image to topic directory"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except IOError as e:
        logging.error(f"Failed to open image: {e}")
        sys.exit(1)

    topic_folder = selected_topic.replace(" ", "_").lower()
    image_directory = os.path.join("images", topic_folder)
    os.makedirs(image_directory, exist_ok=True)
    image_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join(image_directory, image_filename)
    image.save(image_path)

    logging.info(f"Image saved to {image_path}")


@retry(Exception, delay=1, backoff=2, tries=3)
def generate_image(image_prompt):
    """Generate an image from the prompt."""
    api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    hf_token = os.environ["HF_TOKEN"]
    headers = {"Authorization": f"Bearer {hf_token}"}

    def query(payload):
        response = requests.post(api_url, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            logging.error(f"HTTP error occurred: {e}")
            sys.exit(1)
        return response.content

    logging.info("Generating image...")
    generation_params = {
        "inputs": image_prompt,
        "parameters": {
            "num_inference_steps": 50,
            "guidance_scale": 3.5,
            "height": 1024,
            "width": 768,
        },
    }

    start_time = time.time()
    image_bytes = query(generation_params)
    elapsed_time = time.time() - start_time
    logging.info(f"Image generated in {elapsed_time:.2f} seconds.")

    return image_bytes


@retry(Exception, delay=1, backoff=2, tries=3)
def upscale_image(image_bytes):
    # Create a temporary file to save the input image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_input:
        temp_input_path = temp_input.name
        Image.open(io.BytesIO(image_bytes)).save(temp_input_path)

    # Initialize the Gradio client
    client = Client("LuxOAI/AuraUpscale")

    # Perform the prediction (upscaling)
    _, after = client.predict(
        input_image=handle_file(temp_input_path),
        api_name="/process_image",
    )
    # Read the upscaled image as bytes
    with open(after, "rb") as upscaled_file:
        upscaled_image_bytes = upscaled_file.read()
    return upscaled_image_bytes


if __name__ == "__main__":
    main()
