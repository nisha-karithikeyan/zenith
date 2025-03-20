# Automation.py

## Import required libraries
from AppOpener import close, open as appopen  # Import functions to open and close apps.
from webbrowser import open as webopen  # Import web browser functionality.
from pywhatkit import search, playonyt  # Import functions for Google search and YouTube playback.
from dotenv import dotenv_values  # Import dotenv to manage environment variables.
from bs4 import BeautifulSoup  # Import BeautifulSoup for parsing HTML content.
from rich import print  # Import rich for styled console output.
from groq import Groq  # Import Groq for AI chat functionalities.
import webbrowser  # Import webbrowser for opening URLs.
import subprocess  # Import subprocess for interaction with the system.
import requests  # Import requests for making HTTP requests.
import keyboard  # Import keyboard for keyboard-related actions.
import asyncio  # Import asyncio for asynchronous programming.
import os  # Import os for operating system functionalities.

# Load environment variables from the .env file.
env_vars = dotenv_values('.env')
GroqAPIKey = env_vars.get("GroqAPIKey")  # Retrieve the Groq API key.

# Define CSS classes for parsing specific elements in HTML content.
classes = ['Zclount', 'ngKic', 'LIKOO SY/Tric', 'Zelcm', 'gst vk_bk fzwxbb twbrmf', 'pclqee', 'tr-Data-text tr-text-small tr-ta',
    "IZGrede", "OSMB6d LIKOO", "v1zv5d", "webanswers-webanswers_table__webanswers-table", "GD040 1104BB gstt", "sXL40b",
    "UMrKe", "VQF4g", "qv3mpc", "kno-deesc", "SPZzbb"]

# Define a user-agent for mixing web requests.
useragent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (ONTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'

# Initialize the Groq client with the API key.
client = Groq(api_key=GroqAPIKey)

# Predefined professional responses for user interactions.
professional_responses = [
    "Your satisfaction is my top priority; feel free to reach out if there's anything else I can help you with.",
    "I'm at your service for any additional questions or support you may need-don't hesitate to ask.",
]

# List to store chatbot messages.
messages = []

# System message to provide content to the chatbot.
SystemChatBot = [{"role": "system", "content": f"Hello, I am {os.environ['Username']}, You're a content writer. You have to write content like letters, articles, and more."}]

# Function to perform a Google search.
def GoogleSearch(Topic):
    search(Topic)  # Use pywhatkit's search function to perform a Google search.
    return True  # Indicate success.

#GoogleSearch("Nisha")

# Function to generate content using AI and save it to a file.
def Content(Topic):
    Topic = Topic.replace("Content ", "")  # Remove "Content " from the topic.
    ContentByAI = ContentWriterAI(Topic)  # Generate content using AI.

    # Save the generated content to a text file.
    with open(rf"Data\{Topic.lower().replace(' ', '')}.txt", "w", encoding="utf-8") as file:
        file.write(ContentByAI)  # Write the content to the file.
        file.close()

    OpenNotepad(rf"Data\{Topic.lower().replace(' ', '')}.txt")  # Open the file in Notepad.
    return True  # Indicate success.

# Function to open a file in Notepad.
def OpenNotepad(file):
    default_text_editor = 'notepad.exe'  # Default text editor.
    subprocess.Popen([default_text_editor, file])  # Open the file in Notepad.

# Function to generate content using the AI chatbot.
def ContentWriterAI(prompt):
    messages.append({'role': 'user', 'content': prompt})  # Add the user's prompt to messages.

    completion = client.chat.completions.create(
        model='mixtral-8x7b-32768',  # Specify the AI model.
        messages=SystemChatBot + messages,  # Include system instructions and chat history.
        max_tokens=2048,  # Limit the maximum tokens in the response.
        temperature=0.7,  # Adjust response randomness.
        top_p=1,  # Use nucleus sampling for response diversity.
        stream=True,  # Enable streaming response.
        stop=None  # Allow the model to determine stopping conditions.
    )

    Answer = ""  # Initialize an empty string for the response.

    # Process streamed response chunks.
    for chunk in completion:
        if chunk.choices[0].delta.content:  # Check for content in the current chunk.
            Answer += chunk.choices[0].delta.content  # Append the content to the answer.

    Answer = Answer.replace("s/s", "")  # Remove unwanted tokens from the response.
    messages.append({'role': 'assistant', 'content': Answer})  # Add the AI's response to messages.
    return Answer

# Function to search for a topic on YouTube.
def YouTubeSearch(Topic):
    UrlsSearch = f"https://www.youtube.com/results?search_query={Topic}"  # Construct the YouTube search URL.
    webbrowser.open(UrlsSearch)  # Open the search URL in a web browser.
    return True  # Indicate success.

# Function to play a video on YouTube.
def PlayYouTube(query):
    playonyt(query)  # Use pywhatkit's playonyt function to play the video.
    return True  # Indicate success.

# Function to open an application or a relevant webpage.
def OpenApp(app, sess=requests.session()):
    try:
        appopen(app, match_closest=True, output=True, throw_error=True)  # Attempt to open the app.
        return True  # Indicate success.
    except:
        # Nested function to extract links from HTML content.
        def extract_links(html):
            if html is None:
                return []
            soup = BeautifulSoup(html, 'html.parser')  # Parse the HTML content.
            links = soup.find_all('a', {'jsname': 'UMckMb'})  # Find relevant links.
            return [link.get('href') for link in links]  # Return the links.

        # Nested function to perform a Google search and retrieve HTML.
        def search_google(query):
            url = f"https://www.google.com/search?q={query}"  # Construct the Google search URL.
            headers = {"User-Agent": useragent}  # Use the predefined user-agent.
            response = sess.get(url, headers=headers)  # Perform the GET request.

            if response.status_code == 200:
                return response.text  # Return the HTML content.
            else:
                print("Failed to retrieve search results.")  # Print an error message.
                return None

        html = search_google(app)  # Perform the Google search.

        if html:
            link = extract_links(html)[0]  # Extract the first link from the search results.
            webopen(link)  # Open the link in a web browser.

        return True  # Indicate success.

# Function to close an application.
def CloseApp(app):
    if "chrome" in app:
        pass  # Skip if the app is Chrome.
    else:
        try:
            close(app, match_closest=True, output=True, throw_error=True)  # Attempt to close the app.
            return True  # Indicate success.
        except:
            return False  # Indicate failure.
       
CloseApp("outlook")

# Function to execute system-level commands.
def System(command):
    # Nested function to mute the system volume.
    def mute():
        keyboard.press_and_release("volume mute")  # Simulate the mute key press.

    # Nested function to unmute the system volume.
    def unmute():
        keyboard.press_and_release("volume mute")  # Simulate the unmute key press.

    # Nested function to increase the system volume.
    def volume_up():
        keyboard.press_and_release("volume up")  # Simulate the volume up key press.

    # Nested function to decrease the system volume.
    def volume_down():
        keyboard.press_and_release("volume down")  # Simulate the volume down key press.

    # Execute the appropriate command.
    if command == "mute":
        mute()
    elif command == "unmute":
        unmute()
    elif command == "volume up":
        volume_up()
    elif command == "volume down":
        volume_down()

    return True  # Indicate success.

# Asynchronous function to translate and execute user commands.
async def TranslateAndExecute(commands: list[str]):
    funcs = []  # List to store asynchronous tasks.

    for command in commands:
        if command.startswith("open "):
            if "open it" in command:  # Ignore "open it" commands.
                pass
            elif "open file" == command:  # Ignore "open file" commands.
                pass
            else:
                fun = asyncio.to_thread(OpenApp, command.removeprefix("open "))  # Schedule app opening.
                funcs.append(fun)

        elif command.startswith("close "):
            fun = asyncio.to_thread(CloseApp, command.removeprefix("close "))  # Schedule app closing.
            funcs.append(fun)

        elif command.startswith("play "):
            fun = asyncio.to_thread(PlayYouTube, command.removeprefix("play "))  # Schedule YouTube playback.
            funcs.append(fun)

        elif command.startswith("content "):
            fun = asyncio.to_thread(Content, command.removeprefix("content "))  # Schedule content creation.
            funcs.append(fun)

        elif command.startswith("google search "):
            fun = asyncio.to_thread(GoogleSearch, command.removeprefix("google search "))  # Schedule Google search.
            funcs.append(fun)

        elif command.startswith("youtube search "):
            fun = asyncio.to_thread(YouTubeSearch, command.removeprefix("youtube search "))  # Schedule YouTube search.
            funcs.append(fun)

        elif command.startswith("system "):
            fun = asyncio.to_thread(System, command.removeprefix("system "))  # Schedule system commands.
            funcs.append(fun)

        else:
            print(f"No Function Found for {command}")  # Print an error for unrecognized commands.

    results = await asyncio.gather(*funcs)  # Execute all tasks concurrently.
    for result in results:  # Process the results.
        if isinstance(result, str):
            yield result
        else:
            yield result

# Asynchronous function to automate command execution.
async def Automation(commands: list[str]):
    async for result in TranslateAndExecute(commands):  # Translate and execute commands.
        pass
    return True  # Indicate success.

#if __name__ == "__main__":
#     asyncio.run(Automation(["close all apps"]))