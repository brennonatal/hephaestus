import logging

from config import GUIDE, IDEAS
from image import generate_image
from setup import (
    get_batch_size,
    get_upscale_factor,
    get_user_request,
    get_user_topic,
    setup_logging,
    validate_api_keys,
)
from utils import save_image
from workflow import get_workflow

if __name__ == "__main__":

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

    # Step 6: Get agentic workflow
    graph = get_workflow()

    # Step 7: Initialize list to store image paths
    image_paths = []

    # Step 8: Loop for batch_size times
    for i in range(1, batch_size + 1):
        logging.info(f"Processing image {i}/{batch_size}...")
        # Invoke the workflow
        image_prompt_data = graph.invoke(
            {
                "guide": GUIDE,
                "theme": selected_topic,
                "instructions": topic_instructions,
                "request": user_request,
            }
        )

        # Extract the final prompt
        image_prompt = image_prompt_data["final_prompt"].strip()
        logging.info(f"Generated image prompt:\n{image_prompt}")
        # Generate the image
        image = generate_image(image_prompt, upscale_factor)

        # Save the image
        image_path = save_image(selected_topic, image)
        image_paths.append(image_path)

    # Step 9: Log all image paths
    logging.info("Batch generation completed. Image paths:")
    for path in image_paths:
        logging.info(path)
