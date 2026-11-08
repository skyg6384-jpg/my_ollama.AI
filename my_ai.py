"""
MY OWN AI - a simple chat script that talks to your local Ollama model.

What this does:
1. Gives your AI a name and personality (edit the SYSTEM_PROMPT below)
2. Remembers your conversation even after you close and reopen it
   (saved in a file called memory.json, next to this script)
3. Lets you chat with it right here in this window

HOW TO RUN THIS:
1. Make sure Ollama is installed and you've run "ollama run llama3.2" at least once
2. Install one library this script needs by typing in cmd:
       pip install requests
3. Run this script by typing:
       python my_ai.py
"""

import requests
import json
import os
from pypdf import PdfReader

# ---------------------------------------------------------
# STEP 1: Customize your AI here
# ---------------------------------------------------------
AI_NAME = "Nova"  # <-- change this to whatever name you want

SYSTEM_PROMPT = f"""You are {AI_NAME}, a helpful personal AI assistant
running locally on the user's laptop. You are friendly, a little
witty, and you remember past conversations.

IMPORTANT STYLE RULES:
- Keep replies SHORT by default (2-5 sentences), like a real chat,
  not a textbook chapter.
- If asked to teach or explain something big (like a school subject),
  do NOT dump the whole topic at once. Teach ONE small piece, check
  if the user understood, then continue only when they say to.
- Only go long and detailed if the user explicitly asks for "detail",
  "explain more", or similar.
- Ask a follow-up question when it helps keep the conversation going."""

MODEL = "llama3.2"        # the model we downloaded with Ollama
MEMORY_FILE = "memory.json"  # where past conversations get saved


# ---------------------------------------------------------
# STEP 2: Load memory if it exists, otherwise start fresh
# ---------------------------------------------------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # first time running - start with just the personality instructions
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)


# ---------------------------------------------------------
# STEP 3: Send a message to the local AI model and get a reply
# This version STREAMS the reply, printing it word by word as
# it's generated, instead of waiting for the whole thing.
# ---------------------------------------------------------
def ask_ai(messages):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": MODEL,
            "messages": messages,
            "stream": True,  # <-- this turns streaming on
        },
        stream=True,
    )

    full_reply = ""
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        piece = chunk.get("message", {}).get("content", "")
        print(piece, end="", flush=True)  # print as it arrives, no newline
        full_reply += piece
        if chunk.get("done"):
            break

    print()  # newline once the reply is finished
    return full_reply


# ---------------------------------------------------------
# STEP 4: The actual chat loop
# ---------------------------------------------------------
def main():
    messages = load_memory()

    print(f"--- Chatting with {AI_NAME} ---")
    print("Commands: 'quit' to exit | '/clear' to wipe memory")
    print("          '/read filename.txt' or '/read file.pdf' to have Nova read a file\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit"]:
            print(f"\n{AI_NAME}: See you later!")
            break

        if user_input.lower() in ["/clear", "/reset"]:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            save_memory(messages)
            print(f"\n{AI_NAME}: Memory wiped. Starting fresh!\n")
            continue

        # /read lets you point Nova at a file, e.g:  /read notes.txt  or  /read report.pdf
        if user_input.lower().startswith("/read "):
            filename = user_input[6:].strip()  # everything after "/read "
            if not os.path.exists(filename):
                print(f"\n{AI_NAME}: I can't find a file called '{filename}'. "
                      f"Make sure it's in the same folder as my_ai.py, "
                      f"or type the full path.\n")
                continue

            try:
                if filename.lower().endswith(".pdf"):
                    reader = PdfReader(filename)
                    file_content = ""
                    for page in reader.pages:
                        file_content += page.extract_text() + "\n"
                else:
                    with open(filename, "r", encoding="utf-8") as f:
                        file_content = f.read()
            except Exception as e:
                print(f"\n{AI_NAME}: I couldn't open that file ({e}).\n")
                continue

            # keep things fast: don't send a huge file entirely if it's massive
            if len(file_content) > 12000:
                file_content = file_content[:12000] + "\n...(file was long, this is the beginning of it)"

            # hand the file content to the AI along with a note about it
            user_input = (
                f"I'm sharing a file called '{filename}' with you. "
                f"Here is its content:\n\n{file_content}\n\n"
                f"Please read it and let me know you've got it, "
                f"and briefly say what it's about."
            )
            print(f"\n(Nova is reading {filename}...)\n")

        # add the user's message to the conversation
        messages.append({"role": "user", "content": user_input})

        # get the AI's reply (this prints itself, word by word, as it streams)
        print(f"\n{AI_NAME}: ", end="", flush=True)
        reply = ask_ai(messages)
        print()

        # add the AI's reply to the conversation too
        messages.append({"role": "assistant", "content": reply})

        # save everything so it remembers next time
        save_memory(messages)


if __name__ == "__main__":
    main()