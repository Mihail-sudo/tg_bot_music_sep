import logging
import os
import asyncio

from src.bot.bot import MusicBot
from src.llm.llm import OllamaLLMService

from dotenv import load_dotenv
load_dotenv()


TOKEN = os.environ["TELEGRAM_TOKEN"]
API_KEY = os.environ["API_KEY"]
LLM_URL = os.environ["LLM_URL"]


async def main():
    llm = OllamaLLMService(model_name="llama3.2:3b", ollama_base_url=LLM_URL, api_key=API_KEY)
    bot = MusicBot(token=TOKEN, llm_model=llm)
    await bot.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
