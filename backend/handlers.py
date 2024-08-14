"""
Source file for connecting to Telegram and process incoming messages.
"""
from typing import Callable, Coroutine, Any
from enum import Enum, auto
from telegram import Update, Message
from telegram.ext import CallbackContext, ConversationHandler

class STATE(Enum):
    """
    Define the state enum to represent the current processing state of the chatbot.
    Currently not doing anything special.

    Ref: https://docs.python-telegram-bot.org/en/stable/examples.conversationbot2.html (Diagram)

    """
    WAITING = auto()

def message_check(
        func: Callable[[Message, CallbackContext], Coroutine]
    ) -> Callable[[Update, CallbackContext], Coroutine]:
    """
    Python decorator to perform check "Message is not None" instead of doing it in every function
    """
    async def wrapper(update: Update, context: CallbackContext) -> Any:
        __return: Any = None
        if update.message is not None:
            __return = await func(update.message, context)
        else:
            print("Message is None.")
        return __return
    return wrapper

@message_check
# pylint: disable=unused-argument
async def start_command(message: Message, context: CallbackContext) -> STATE:
    """
    Telegram bot /start command handler.

    Args:
        - message (telegram.Message): Message property from Update object.
        - context (telegram.ext.CallbackContext): Irrelevant for now.

    Returns
        STATE: The WAITING state of the bot.  
    """
    await message.reply_text("Hello! Let's start.")
    return STATE.WAITING


@message_check
# pylint: disable=unused-argument
async def cancel_command(message: Message, context: CallbackContext) -> int:
    """
    Telegram bot /cancel command handler.

    Args:
        - message telegram.Message: Message property from Update object.
        - context telegram.ext.CallbackContext: Irrelevant for now.

    Returns
    int: The turn off state of the bot.
    """
    await message.reply_text("Bye! I hope we can talk again some day.")
    return ConversationHandler.END
