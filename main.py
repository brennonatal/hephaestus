import random
import getpass
import os
from prompts import GUIDE, IDEAS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Validate or prompt for the Groq API key
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Step 1: Select a random topic
topics = list(IDEAS.keys())
selected_topic = random.choice(topics)

# Step 2: Get the topic instructions
topic_instructions = IDEAS[selected_topic]

# Step 3: Set up the Groq LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Step 4: Create a ChatPromptTemplate with separate system and human messages
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are an image prompt generator specialized in FLUX models. Your task is to create detailed and effective image prompts based on the user's topic and instructions.  When generating the prompt, include all necessary details as per the instructions, and ensure that your output is ONLY the image prompt without any additional explanations or text!\n\n{GUIDE}",
        ),
        (
            "human",
            f"Please create a detailed image prompt for the following topic:\n\n{selected_topic}\n\n{topic_instructions}",
        ),
    ]
)

# Step 5: Chain the prompt with the LLM
chain = prompt | llm

# Step 6: Invoke the chain with the topic and instructions
ai_msg = chain.invoke({"topic": selected_topic, "instructions": topic_instructions})

# Step 7: Print the output
print(ai_msg.content)
