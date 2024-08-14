"""
Driver code file
"""
# import os

# from dotenv import load_dotenv
from telegram.ext import (Application, CommandHandler, ConversationHandler,
                          MessageHandler, filters)

from backend.const import API_TOKEN
from backend.handlers import (STATE, cancel_command, message_check,
                              start_command)

from conv_chain import get_langchain_chain

import subprocess
import threading
import time
import signal
import sys


conversational_rag_chain = get_langchain_chain()


def start_ollama_server():
    # Start the ollama server in a new process
    process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def cleanup():
        print("Cleaning up...")
        if process.poll() is None:  # If the process is still running
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        sys.exit(0)

    # Ensure the process is terminated on exit
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())

    # Function to print server output
    def print_output(process):
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                print(output.strip().decode('utf-8'))
            time.sleep(1)

    # Start a thread to print server output
    thread = threading.Thread(target=print_output, args=(process,))
    thread.daemon = True  # Daemon thread will exit when the main program does
    thread.start()

    return process

@message_check
async def handle_response(message, context) -> STATE:
    """
    The current main response handler. This function checks the chat type 
    and asking the response from the Language Model.

    Args:
        - message (telegram.Message): Message property from Update object.
        - context (telegram.ext.CallbackContext): Context object passed by 
        the python-telegram-bot library.

    Returns
        STATE: The WAITING state of the bot.  
    """
    # pylint: disable=invalid-name
    def ask_LLM(query: str) -> str:
        response = conversational_rag_chain.invoke(
            {"input": query} #,
            #config={
            #    "configurable": {"session_id": "abc123"}
            #},  # constructs a key "abc123" in `store`.
        )

        print('\n' * 5)
        print(f"Question: {query}", '\n' * 2)
        for i in response['context']:
            print (i, end='\n' * 3)

        return response['answer']

    if message.chat.type in ["group", "supergroup"] and message.entities:
        for entity in message.entities:
            if entity.type == "mention" and message.text is not None:
                # Extract the mentioned username
                mentioned_username = message.text[
                    entity.offset:entity.offset + entity.length
                ] # type: ignore
                if mentioned_username == f"@{context.bot.username}":
                    input_str: str = message.text.replace(f"@{context.bot.username}", "")
                    print(input_str)
                    await message.reply_text(ask_LLM(input_str))
                    return STATE.WAITING
    else:
        await message.reply_text(ask_LLM(message.text))  # type: ignore

    return STATE.WAITING

def main() -> None:
    """
    main() routine
    """
        # Run the installation script
    subprocess.run("curl https://ollama.ai/install.sh | sh", shell=True, check=True)

    # Start the server
    process = start_ollama_server()

    try:
        # Place your coding logic here (before pulling the model)
        # -------------------------------------
        # Example: Perform some initialization or setup
        print("Performing pre-pull setup...")
        # Your logic here
        # -------------------------------------

        # Pull the model in parallel (non-blocking)
        subprocess.run("ollama pull llama3.1:8b", shell=True, check=True)

        # Place your coding logic here (after pulling the model)
        print("\nPolling started...")
        application = Application.builder().token(API_TOKEN).build()

        # Command and handlers mapping
        conversation_handler = ConversationHandler(
            entry_points=[CommandHandler("start", start_command)],
            states={
                STATE.WAITING: [MessageHandler(
                    filters.TEXT & ~filters.COMMAND, handle_response)]
            },
            fallbacks=[CommandHandler("cancel", cancel_command)]
        )

        # Add handler to application and run
        application.add_handler(conversation_handler)
        application.run_polling()
        # Example: Perform some operations after the model is pulled
        print("Performing post-pull operations...")
        # Your logic here
        # -------------------------------------

    finally:
        # Ensure the server is terminated
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()




if __name__ == "__main__":
    main()
