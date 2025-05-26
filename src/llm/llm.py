from .llm_abs import LLMService, AnswerDTO, QuestionDTO
from typing import cast
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


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
        self._chain = prompt | llm

    async def execute(self, data: QuestionDTO) -> AnswerDTO:
        response = await self._chain.ainvoke({
            "question": data.text,
            "history": [(message.role, message.text) for message in data.history]
        })
        response = cast(AIMessage, response)
        return AnswerDTO(
            text=response.content,
            used_tokens=response.usage_metadata.get("total_tokens", 0)
        )