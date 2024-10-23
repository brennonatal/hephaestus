import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from retry import retry

from config import GUIDE, IDEAS
from image import generate_image
from models import ImagePrompt
from setup import (
    get_batch_size,
    get_upscale_factor,
    get_user_request,
    get_user_topic,
    setup_groq_llm,
    setup_logging,
    validate_api_keys,
)
from utils import save_image


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

        # Save the image
        image_path = save_image(selected_topic, image)
        image_paths.append(image_path)

    # Step 11: Log all image paths
    logging.info("Batch generation completed. Image paths:")
    for path in image_paths:
        logging.info(path)


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


if __name__ == "__main__":
    main()
