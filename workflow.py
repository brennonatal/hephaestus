from langgraph.graph import StateGraph

from agent.nodes import prompt_generator
from agent.state import OutputState, State


def get_workflow():
    # Define a new graph
    workflow = StateGraph(State, output=OutputState)

    # Define the two nodes we will cycle between
    workflow.add_node("prompt_generator", prompt_generator)
    # workflow.add_node("action", tool_node)

    workflow.set_entry_point("prompt_generator")
    workflow.set_finish_point("prompt_generator")

    return workflow.compile(debug=False)
