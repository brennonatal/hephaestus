from typing import TypedDict


class State(TypedDict):
    guide: str
    theme: str
    instructions: str
    request: str
    height: int
    width: int
    upscale_factor: int
    examples: str
    final_prompt: str


class OutputState(TypedDict):
    final_prompt: str
