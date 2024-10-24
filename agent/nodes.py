from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from agent.tools import tools


@lru_cache(maxsize=4)
def _get_model():
    try:
        model = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.8,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        model = model.bind_tools(tools)
        return model
    except Exception as e:
        print(f"Error getting model: {e}")
        return None


# Define the function that calls the model
def prompt_generator(state):
    guide = state["guide"]
    theme = state["theme"]
    instructions = state["instructions"]
    request = state["request"]

    model = _get_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an image prompt generator specialized in FLUX models. "
                    "Your task is to create detailed and effective image prompts based on the user's topic, instructions, and specific requests. "
                    "No function tool calling is available. "
                    "Output ONLY the final prompt and nothing else!\n\n"
                    "{guide}"
                ),
            ),
            (
                "user",
                (
                    "Please create a detailed image prompt for the following topic:\n\n"
                    "{theme}\n\n{instructions}\n\n"
                    "Additional user request (if any):\n\n{request}"
                    "\n\nOutput ONLY the final prompt following the example and nothing else!\n\n"
                ),
            ),
        ]
    )

    chain = prompt | model
    response = chain.invoke(
        {
            "guide": guide,
            "theme": theme,
            "instructions": instructions,
            "request": request,
        }
    )
    state["final_prompt"] = response.content.strip()
    return state


# Define the function to execute tools
tool_node = ToolNode(tools)
