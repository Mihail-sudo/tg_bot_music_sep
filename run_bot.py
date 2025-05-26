import os
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from aiogram import F


from src.llm.llm import OllamaLLMService, QuestionDTO


TOKEN = os.environ["TELEGRAM_TOKEN"]

llm = OllamaLLMService(model_name="llama3.2:3b", ollama_base_url="http://localhost:11434")


dp = Dispatcher(storage=MemoryStorage())

@dp.message(F.text)
async def handle_music_query(message: Message):
    user_query = message.text.strip()
    await message.reply("Подожди секунду... думаю над ответом 🧠")

    bot_response = await llm.execute(QuestionDTO(text=user_query, history=[]))

    for chunk in [bot_response.text[i:i+4096] for i in range(0, len(bot_response.text), 4096)]:
        await message.reply(bot_response.text)


async def main():
    bot = Bot(token=TOKEN)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())