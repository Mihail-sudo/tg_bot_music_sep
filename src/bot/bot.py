from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram import F
from aiogram.fsm.context import FSMContext
from src.llm.llm_abs import QuestionDTO, MessageDTO, LLMService


from dotenv import load_dotenv
load_dotenv()


class MusicBot:
    def __init__(self, token: str, llm_model: LLMService):
        self.bot = Bot(token=token)
        self.llm = llm_model

        self.dp = Dispatcher()
        self.dp.message.register(self.handle_music_query, F.text)

    async def run(self):
        await self.dp.start_polling(self.bot)

    async def handle_music_query(self, message: Message, state: FSMContext):
        user_query = message.text.strip()

        data = await state.get_data()
        history = data.get("history", [])

        await self.bot.send_chat_action(chat_id=message.from_user.id, action='typing')
        bot_response = await self.llm.execute(QuestionDTO(text=user_query, history=history))

        full_text = ''
        for chunk in [bot_response.text[i:i+4096] for i in range(0, len(bot_response.text), 4096)]:
            full_text += chunk
            await message.reply(chunk)

        history.append(MessageDTO(role='human', text=user_query))
        history.append(MessageDTO(role='assistant', text=full_text))
        await state.update_data(history=history)
