import base64
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
from typing import Any, Dict

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
    """Main function to generate images based on user-selected or random topic."""
    setup_logging()
    validate_api_keys()

    # Step 1: Get the topic from the user
    selected_topic = get_user_topic(IDEAS)

    # Step 2: Get the topic instructions
    topic_instructions = IDEAS[selected_topic]

    # Step 3: Ask the user for any specific requests
    user_request = get_user_request()

    # Step 4: Ask the user for batch size
    batch_size = get_batch_size()

    # Step 5: Ask the user upscaling preferences
    upscale_factor = get_upscale_factor()

    # Step 6: Set up the Groq LLM
    llm = setup_groq_llm()

    # Wrap the LLM with structured output
    structured_llm = llm.with_structured_output(ImagePrompt)

    # Get the JSON schema
    schema = ImagePrompt.model_json_schema()

    # Step 7: Create a ChatPromptTemplate with system and human messages
    prompt = create_chat_prompt_template()

    # Step 8: Chain the prompt with the structured LLM
    chain = prompt | structured_llm

    # Step 9: Initialize list to store image paths
    image_paths = []

    # Step 10: Loop for batch_size times
    for i in range(1, batch_size + 1):
        logging.info(f"Processing image {i}/{batch_size}...")
        # Invoke the chain
        image_prompt_data = generate_image_prompt(
            chain, GUIDE, selected_topic, topic_instructions, schema, user_request
        )

        # Extract the final prompt
        image_prompt = image_prompt_data.final_prompt.strip()
        logging.info(f"Generated image prompt:\n{image_prompt}")

        # Generate the image
        image = generate_image(image_prompt, upscale_factor)

        # Upscale the image
        # upscaled_image_bytes = upscale_image(image_bytes)

        # Save the image
        image_path = save_image(selected_topic, image)
        image_paths.append(image_path)

    # Step 11: Log all image paths
    logging.info("Batch generation completed. Image paths:")
    for path in image_paths:
        logging.info(path)


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
    if "INFERENCE_ENDPOINT" not in os.environ:
        os.environ["INFERENCE_ENDPOINT"] = getpass.getpass(
            "Enter your Inference Endpoint: "
        )


def get_user_topic(ideas_dict):
    """Prompt the user to select a topic or choose a random one."""
    topics = list(ideas_dict.keys())
    print("Available topics:")
    for idx, topic in enumerate(topics, 1):
        print(f"{idx}. {topic}")

    user_input = input(
        "\nEnter the number of the topic you want to select or press Enter for a random topic: "
    )

    if user_input.strip() == "":
        selected = random.choice(topics)
        logging.info(f"Randomly selected topic: {selected}")
    else:
        try:
            selection = int(user_input)
            if 1 <= selection <= len(topics):
                selected = topics[selection - 1]
                logging.info(f"User selected topic: {selected}")
            else:
                logging.error("Invalid selection. Selecting a random topic.")
                selected = random.choice(topics)
                logging.info(f"Randomly selected topic: {selected}")
        except ValueError:
            logging.error("Invalid input. Selecting a random topic.")
            selected = random.choice(topics)
            logging.info(f"Randomly selected topic: {selected}")

    return selected


def get_user_request():
    """Prompt the user for any specific requests to include in the image prompt."""
    user_request = input(
        "\nEnter any specific instructions or details you want to include (or press Enter to skip): "
    ).strip()
    if user_request:
        logging.info(f"User's specific request: {user_request}")
    else:
        logging.info("No specific request provided by the user.")
    return user_request


def get_batch_size():
    """Prompt the user to enter the number of images to generate."""
    while True:
        user_input = input(
            "\nEnter the number of images you want to generate (default is 1): "
        ).strip()
        if user_input == "":
            batch_size = 1
            logging.info("Defaulting to generating 1 image.")
            break
        try:
            batch_size = int(user_input)
            if batch_size >= 1:
                logging.info(f"Batch size selected: {batch_size}")
                break
            else:
                logging.warning("Please enter a positive integer.")
        except ValueError:
            logging.warning("Invalid input. Please enter a positive integer.")
    return batch_size


def get_upscale_factor():
    """Prompt the user to enter the upscaling factor for the images."""
    while True:
        user_input = input(
            "\nEnter the upscaling factor (default is 1, maximum is 8): "
        ).strip()
        if user_input in ["", "1"]:
            upscale_factor = 0
            logging.info("Defaulting to upscaling factor of 1.")
            break
        try:
            upscale_factor = int(user_input)
            if upscale_factor in [2, 4, 8]:
                logging.info(f"Selected upscaling factor of {upscale_factor}.")
                break
            else:
                logging.warning("Unsupported upscale factor. Choose from 2, 4, or 8.")
        except ValueError:
            logging.warning(
                "Invalid input. Please enter a numeric value between 1 and 8."
            )
    return upscale_factor


def setup_groq_llm():
    """Initialize the Groq LLM with specified parameters."""
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.8,
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
                    "Your task is to create detailed and effective image prompts based on the user's topic, instructions, and specific requests. "
                    "No function tool calling is available. "
                    "Output the final prompt in JSON format matching the following schema:\n\n"
                    "{schema}\n\n"
                    "Ensure that your output is ONLY the JSON object without any additional explanations or text!\n\n{GUIDE}"
                ),
            ),
            (
                "human",
                (
                    "Please create a detailed image prompt for the following topic:\n\n"
                    "{topic}\n\n{instructions}\n\n"
                    "Additional user request (if any):\n\n{user_request}"
                ),
            ),
        ]
    )


@retry(Exception, delay=1, backoff=2, tries=3)
def generate_image_prompt(chain, guide, topic, instructions, schema, user_request):
    """Generate an image prompt using the structured LLM chain."""
    logging.info(f"Requesting image prompt for topic: '{topic}'...")
    try:
        image_prompt_data = chain.invoke(
            {
                "GUIDE": guide,
                "schema": json.dumps(schema),
                "topic": topic,
                "instructions": instructions,
                "user_request": user_request,
            }
        )
    except Exception as e:
        logging.error(f"Error generating image prompt: {e}")
        raise e

    return image_prompt_data


@retry(Exception, delay=1, backoff=2, tries=3)
def generate_image(image_prompt, upscale_factor=0) -> Image.Image:
    """Generate an image from the prompt, optionally upscaling it."""
    api_url = os.environ["INFERENCE_ENDPOINT"]
    hf_token = os.environ["HF_TOKEN"]
    headers = {"Authorization": f"Bearer {hf_token}"}

    def encode_image_to_base64(image: Image.Image) -> str:
        """
        Encode a PIL Image to a base64 string.

        :param image: PIL Image.
        :return: Base64 encoded string of the image.
        """
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return image_base64
        except Exception as e:
            logging.error(f"Failed to encode image: {e}")
            raise

    def decode_base64_to_image(image_base64: str) -> Image.Image:
        """
        Decode a base64 string to a PIL Image.

        :param image_base64: Base64 encoded image string.
        :return: PIL Image object.
        """
        try:
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception as e:
            logging.error(f"Failed to decode image: {e}")
            raise

    def query(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a POST request to the inference endpoint.

        :param payload: The JSON payload for the request.
        :return: JSON response from the server.
        """
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err} - {response.text}")
            raise
        except Exception as err:
            logging.error(f"An error occurred: {err}")
            raise

    logging.info("Generating low-resolution image...")
    # Generate the image at lower resolution
    generation_params = {
        "inputs": image_prompt,
        "num_inference_steps": 50,
        "guidance_scale": 3.5,
        "height": 1024,
        "width": 768,
        # Do not include 'upscale_factor' here
    }

    try:
        # Generate low-resolution image
        response = query(generation_params)
        # Extract the base64 image from the response
        generated_image_b64 = response.get("image", "")
        if not generated_image_b64:
            logging.error("No image found in the response.")
            return None

        # Decode the base64 image
        generated_image = decode_base64_to_image(generated_image_b64)

        if upscale_factor > 0:
            # Upscale the image
            logging.info("Upscaling image...")
            # Encode the generated image to base64
            control_image_b64 = encode_image_to_base64(generated_image)

            # Prepare upscaling payload
            upscaling_params = {
                "inputs": "",  # Empty prompt as per the upscaling example
                "control_image": control_image_b64,
                "upscale_factor": upscale_factor,
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "controlnet_conditioning_scale": 0.6,
                # Heights and widths are handled by the server based on the control image
            }

            # Send upscaling request
            upscaled_response = query(upscaling_params)
            upscaled_image_b64 = upscaled_response.get("image", "")
            if not upscaled_image_b64:
                logging.error("No image found in the upscaling response.")
                return generated_image  # Return the low-res image if upscaling fails

            # Decode the upscaled image
            upscaled_image = decode_base64_to_image(upscaled_image_b64)
            return upscaled_image
        else:
            return generated_image
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        raise e


def save_image(selected_topic, image):
    """Save the image to topic directory and return the image path."""
    topic_folder = selected_topic.replace(" ", "_").lower()
    image_directory = os.path.join("images", topic_folder)
    os.makedirs(image_directory, exist_ok=True)
    image_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join(image_directory, image_filename)
    image.save(image_path)

    logging.info(f"Image saved to {image_path}")
    return image_path


if __name__ == "__main__":
    main()
