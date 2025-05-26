import os
TOKEN = os.environ["TELEGRAM_TOKEN"]


import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from handlers import router


from src.llm.llm import OllamaLLMService, QuestionDTO
from aiogram.types import Message


from aiogram import Dispatcher, F
dp = Dispatcher()


@dp.message(F.text)
async def handle_music_query(message: Message):
    user_query = message.text.strip()
    
    await message.reply("Подожди секунду... думаю над ответом 🧠")
    
    # Получаем ответ от Ollama
    bot_response = await llm.execute(QuestionDTO(text="Hi, how are you?", history=[]))
    
    for chunk in [bot_response[i:i+4096] for i in range(0, len(bot_response), 4096)]:
        await message.reply(chunk)



async def main():
    bot = Bot(token=TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    llm = OllamaLLMService(model_name="llama3.2:3b", ollama_base_url="http://localhost:11434")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
