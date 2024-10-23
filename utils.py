import logging
import os
import uuid


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
