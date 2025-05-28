from .llm_abs import LLMService, AnswerDTO, QuestionDTO
from typing import cast
from langchain_core.messages import AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from .utils import token_counter


class OllamaLLMService(LLMService):
    _MESSAGES = [
        ("system", "You are musician assistant"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]

    def __init__(self, model_name: str, ollama_base_url: str, api_key: str):
        llm = ChatOpenAI(
            model=model_name,
            base_url=f"{ollama_base_url}/v1",
            api_key=api_key
        )
        prompt = ChatPromptTemplate(self._MESSAGES)

        trimmer = trim_messages(
            strategy="last",
            token_counter=token_counter,
            max_tokens=512,
            start_on="human",
            include_system=True,
            allow_partial=False
        )

        self.store = {}
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = InMemoryChatMessageHistory()
            return self.store[session_id]

        chain = prompt | trimmer | llm
        self._chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    async def execute(self, data: QuestionDTO) -> AnswerDTO:
        response = await self._chain.ainvoke({
            "question": data.text,
            "history": [(message.role, message.text) for message in data.history],
        }, config={"configurable": {"session_id": data.user_id}})

        response = cast(AIMessage, response)
        return AnswerDTO(
            text=response.content,
            used_tokens=response.usage_metadata.get("total_tokens", 0)
        )