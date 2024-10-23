import base64
import io
import logging
import os
from typing import Any, Dict

import requests
from PIL import Image
from retry import retry


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
