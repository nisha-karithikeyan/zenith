from groq import Groq  # Importing the Groq library to use its API.
from json import load, dump  # Importing functions to read and write JSON files.
import datetime  # Importing the datetime module for real-time date and time information.
from dotenv import dotenv_values  # Importing dotenv_values to read environment variables

# Load environment variables from the .env file.
env_vars = dotenv_values('.env')

# Retrieve specific environment variables for username, assistant name, and API key.
Username = env_vars.get('Username')
Assistantname = env_vars.get('Assistantname')
GroqAPIKey = env_vars.get('GroqAPIKey')

# Initialize the Groq Client using the provided API key.
client = Groq(api_key=GroqAPIKey)

# Initialize an empty list to store chat messages.
messages = []

# Define a system message that provides context to the AI chatbot about its role and behavior.
System = f'''
Hello, I am {Username}. You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date knowledge.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** Reply in only English, even if the question is in Tamil, reply in English.***
*** Do not provide notes in the output, just answer the question and never mention your training data. ***

If the user asks their name, respond with: "Your name is {Username}."
If the user asks who created you, respond with: "I was created by {Username}, a talented developer and researcher."
'''

# A list of system instructions for the chatbot.
SystemChatBot = [
    {"role": "system", "content": System}
]

# Attempt to load the chat log from a JSON file.
try:
    with open(r'Data/ChatLog.json', 'r') as f:
        messages = load(f)  # Load existing messages from the chat log.
except FileNotFoundError:
    # If the file doesn't exist, create an empty JSON file to store chat logs.
    with open(r'Data/ChatLog.json', 'w') as f:
        dump([], f)

def RealtimeInformation():
    current_date_time = datetime.datetime.now()  
    day = current_date_time.strftime('%A')
    date = current_date_time.strftime('%d')
    month = current_date_time.strftime('%B')
    year = current_date_time.strftime('%Y')
    hour = current_date_time.strftime('%H')
    minute = current_date_time.strftime('%M')
    second = current_date_time.strftime('%S')

    data = f"Please use this real-time information if needed,\nDay: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\nTime: {hour} hours :{minute} minutes :{second} seconds.\n"
    return data

def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer

def ChatBot(Query):
    try:
        with open(r'Data/ChatLog.json', 'r') as f:
            messages = load(f)

        messages.append({"role": "user", "content": f"{Query}"})

        completion = client.chat.completions.create(
            model='llama3-70b-8192',
            messages=SystemChatBot + [{"role": "user", "content": Query}],
            max_tokens=1024,
            temperature=0.7,
            stream=True,
            stop=None
        )

        Answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content

        Answer = Answer.replace('</s>', '')

        messages.append({"role": "assistant", "content": Answer})

        with open(r'Data/ChatLog.json', 'w') as f:
            dump(messages, f, indent=4)

        return AnswerModifier(Answer)

    except Exception as e:
        print(f"Error: {e}")
        with open(r'Data/ChatLog.json', 'w') as f:
            dump([], f, indent=4)
        return ChatBot(Query)

if __name__ == '__main__':
    while True:
        user_input = input('Enter Your Question: ')
        print(ChatBot(user_input))
