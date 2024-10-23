import getpass
import logging
import os
import random

from langchain_groq import ChatGroq


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


def setup_groq_llm():
    """Initialize the Groq LLM with specified parameters."""
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.8,
        max_tokens=None,
        timeout=None,
        max_retries=2,
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
