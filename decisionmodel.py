import cohere
from rich import print
from dotenv import dotenv_values

# Load environment variables
env_vars = dotenv_values(".env")
cohereapikey = env_vars.get("COHERE_API_KEY")
cohereclient = cohere.Client(cohereapikey)

# Only allow general and realtime responses
valid_types = ["general", "realtime"]

# Instruction for categorizing queries
preamble = """
You are a very accurate Decision-Making Model, which decides what kind of a query is given to you.
You will decide whether a query is a 'general' query or a 'realtime' query.
*** Do not answer any query, just decide what kind of query is given to you. ***

-> Respond with 'general (query)' if the query can be answered by a conversational LLM and doesn't require real-time updates. For example:
    - 'who was akbar?' → 'general who was akbar?'
    - 'how to write a good resume?' → 'general how to write a good resume?'
    - 'what is the capital of India?' → 'general what is the capital of India?'

-> Respond with 'realtime (query)' if the query requires real-time data or current events. For example:
    - 'who is the current prime minister of India?' → 'realtime who is the current prime minister of India?'
    - 'what's the weather like in Delhi now?' → 'realtime what's the weather like in Delhi now?'

*** Respond with 'general (query)' if you're unsure or the type is ambiguous. ***
"""

# Chat history examples
ChatHistory = [
    {"role": "User", "message": "how are you?"},
    {"role": "Chatbot", "message": "general how are you?"},
    {"role": "User", "message": "what is the time in new york?"},
    {"role": "Chatbot", "message": "general what is the time in new york?"},
    {"role": "User", "message": "who is the current CEO of Google?"},
    {"role": "Chatbot", "message": "realtime who is the current CEO of Google?"},
]

usermsgs = []

def getdecision(prompt: str = "test"):
    usermsgs.append({"role": "User", "message": prompt})

    stream = cohereclient.chat_stream(
        model='command-r-plus',
        message=prompt,
        temperature=0.3,
        chat_history=ChatHistory,
        prompt_truncation='OFF',
        connectors=[],
        preamble=preamble
    )

    response = ""
    for event in stream:
        if event.event_type == "text-generation":
            response += event.text

    response = response.replace("\n", "").strip()

    # Only allow "general ..." or "realtime ..." responses
    for key in valid_types:
        if response.lower().startswith(key):
            return response

    # Default to general if nothing matches
    return f"general {prompt}"

if __name__ == "__main__":
    while True:
        query = input("Enter your query: ")
        print(getdecision(prompt=query))
