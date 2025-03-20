import cohere # for ai services
from rich import print #terminal op enhances
from dotenv import dotenv_values #load the env variables

#env from .env file
env_vars = dotenv_values(".env")

#retrieve api key
CohereAPIKey =  env_vars.get("CohreAPIKey")

#cohere client using the provided api key
CohereAPIKey = env_vars.get("CohereAPIKey")

#cohere client using the provided api key
co = cohere.Client(api_key=CohereAPIKey)

#list of recognized function keywords for task categorization
funcs = ["exit","general","realtime","open","close",
        "play","generate image","system","content",
        "google search","youtube search","reminder"]

#initialize an empty list to store user msgs
messages = []

#preamble that gi=uides ai model to categorize queries
preamble = """
You are a very accurate Decision-Making Model, which decides what kind of a query is given to you.
You will decide whether a query is a 'general query', a 'realtime' query, or is asking to perform any task or automation like 'open facebook'.
*** Do not answer any query, just decide what kind of query is given to you. ***
-> Respond with 'general ( query )' if a query can be answered by a llm model (conversational ai chatbot) and doesn't require any ui to be opened.
-> Respond with 'realtime ( query )' if a query can not be answered by a llm model (because they don't have realtime data) and requires up to date information from the internet.
-> Respond with 'open (application name or website name)' if a query is asking to open any application like 'open facebook', 'open telegram', 'open application name or website name' if a query is asking to open any application like 'close notepad', 'close facebook', etc. but if the query is asking to open multiple tasks like 'open facebook, telegram and close whatsapp' respond with 'open facebook, open telegram, close whatsapp'.
-> Respond with 'play (song name)' if a query is asking to play any song like 'play afsana by ys', 'play let her go', etc. but if the query is asking to play a song from a specific app like 'play afsana by ys from spotify' respond with 'open spotify, play afsana by ys'.
-> Respond with 'generate image (image prompt)' if a query is requesting to generate a image with given prompt like 'generate image of a lion roaring in jungle'.
-> Respond with 'reminder (datetime with message)' if a query is requesting to set a reminder like 'set a reminder like 'set a reminder at 9:00pm on 25th june for finishing assignment'.
-> Respond with 'system (task name)' if a query is asking to mute, unmute, volume up, volume down, etc. but if the query is asking to do multiple tasks respond with 'system (task name 1), system (task name 2), ...'.
-> Respond with 'content (topic)' if a query is asking to write any type of content like application, codes, emails or anything else about the topic.
-> Respond with 'google search (topic)' if a query is asking to search a specific topic on google but if the query is asking to search multiple topics respond with 'google search (topic 1), google search (topic 2), ...'.
-> Respond with 'youtube search (topic)' if a query is asking to search a specific topic on youtube but if the query is asking to search multiple topics respond with 'youtube search (topic 1), youtube search (topic 2), ...'.
-> Respond with 'exit' if the user is saying goodbye or wants to end the conversation like 'bye jarvis', respond with 'exit'.
-> Respond with 'general (query)' if you can't decide the kind of query or if a query is asking to perform a task which is not mentioned above.
"""


# Define a chat history with predefined user-chatbot interactions for context.
ChatHistory = [
    {"role": "User", "message": "how are you?"},
    {"role": "Chatbot", "message": "general how are you?"},
    {"role": "User", "message": "do you like pizza?"},
    {"role": "Chatbot", "message": "general do you like pizza?"},
    {"role": "User", "message": "open chrome and tell me about mahatma gandhi."},
    {"role": "Chatbot", "message": "open chrome, general tell me about mahatma gandhi."},
    {"role": "User", "message": "open chrome and firefox"},
    {"role": "Chatbot", "message": "open chrome, open firefox"},
    {"role": "User", "message": "what is today's date and by the way remind me that I have..."},
    {"role": "Chatbot", "message": "general what is today's date, reminder 11:00pm 5th aug"},
    {"role": "User", "message": "chat with me."},
    {"role": "Chatbot", "message": "general chat with me."}
]

import cohere  # Library for interacting with the Cohere language model


def FirstLayerDWM(prompt: str = "test"):
    # Add the user's query to the messages list.
    messages = [{"role": "user", "content": f"{prompt}"}]

    # Create a streaming chat session with the Cohere model.
    stream = co.chat_stream(  
        model='command-r-plus',     # Specify the Cohere model to use.
        message=prompt,             # The user's query is passed here.
        temperature=0.7,            # Controls creativity (0.0 = deterministic, 1.0 = highly creative).
        chat_history=ChatHistory,   # Provides the predefined chat history for context.
        prompt_truncation='OFF',    # Ensures the prompt is not truncated.
        connectors=[],              # No additional connectors are used.
        preamble=preamble           # Provides a detailed instruction preamble to guide the model's response.
    )

    # Initialize an empty string to store the generated response.
    response = ""

    # Iterate over events in the stream and capture text generation events.
    for event in stream:  
        if event.event_type == "text-generation":  # Check if the event is a text generation event.
            response += event.text  # Append generated text to the response.

    # Remove newline characters and split responses into individual tasks.
    response = response.replace("\n", "")  # Remove all newline characters from the response.
    response = response.split(".")  # Split the response into separate tasks using periods (".").

    # Strip leading and trailing whitespaces from each task.
    response = [i.strip() for i in response]

    # Initialize an empty list to filter valid tasks.
    temp = []

    # Filter the tasks based on recognized function keywords.
    for task in response:
        for func in funcs:  # 'funcs' is a predefined list of function keywords.
            if task.startswith(func):  # Check if the task starts with a recognized keyword.
                temp.append(task)  # Add valid tasks to the filtered list.

    # Update the response with the filtered list of tasks.
    response = temp

    # Check if "Query" is in the response, recursively call the function for further clarification.
    if "Query" in response:
        newresponse = FirstLayerDWM(prompt=prompt)  # Re-run the model with the same prompt.
        return newresponse  # Return the clarified response.
    else:
        return response  # Return the filtered response.


if __name__ == "__main__":
    # Continuously prompt the user for input and process it.
    while True:
        print(FirstLayerDWM(input(">>> ")))  # Print the categorized response.
