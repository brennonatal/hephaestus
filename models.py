from pydantic import BaseModel, Field


class ImagePrompt(BaseModel):
    """Detailed image prompt for FLUX model."""

    final_prompt: str = Field(description="The final detailed image prompt.")
