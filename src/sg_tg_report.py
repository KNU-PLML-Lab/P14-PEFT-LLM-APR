import os
import sys
from sg_utils import send_telegram_message
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
  # Command line arguments to send a Telegram message
  all_args = sys.argv[1:]
  text = ' '.join(all_args) if all_args else "No message provided"

  send_telegram_message(
    text=text,
    bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
    chat_id=os.getenv('TELEGRAM_CHAT_ID'),
    message_thread_id=os.getenv('TELEGRAM_MESSAGE_THREAD_ID'),
    timeout=10,
    background=False,
    silent=True
  )