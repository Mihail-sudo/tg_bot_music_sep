from .llm_abs import LLMService, AnswerDTO, QuestionDTO
from typing import Callable, Dict, cast
from langchain_core.messages import AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from .utils import token_counter, requires_tool
from .tools import tools


class OllamaLLMService(LLMService):
    def __init__(self, model_name: str, ollama_base_url: str, api_key: str):
        self._model = self._create_model(model_name, ollama_base_url, api_key)
        self._promt = ChatPromptTemplate.from_messages([
            ("system", "You are musician assistant"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])

        self._trimmer = self._create_trimmer()
        self._chain = self._create_chain()

        self._agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You can use tools to find lyrics. Answer should be a full song text."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        self._agent = self._create_agent()

    def _create_model(self, model_name: str, base_url: str, api_key: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_name,
            base_url=f"{base_url}/v1",
            api_key=api_key
        )
    
    def _create_agent(self) -> AgentExecutor:
        agent = create_tool_calling_agent(self._model, tools, self._agent_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def _create_trimmer(self) -> Callable[[list[BaseMessage]], list[BaseMessage]]:
        return trim_messages(
            strategy="last",
            token_counter=token_counter,
            max_tokens=512,
            start_on="human",
            include_system=True,
            allow_partial=False
        )
    
    def _create_chain(self) -> RunnableWithMessageHistory:
        store: Dict[str, BaseChatMessageHistory] = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]

        chain = self._promt | self._trimmer | self._model

        return RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    async def execute(self, data: QuestionDTO) -> AnswerDTO:
        if requires_tool(data.text):
            print()
            return await self._run_agent(data)
        else:
            return await self._run_model(data)

    async def _run_model(self, data: QuestionDTO) -> AnswerDTO:
        response = await self._chain.ainvoke({
            "question": data.text,
            "history": [(message.role, message.text) for message in data.history],
        }, config={"configurable": {"session_id": data.user_id}})

        response = cast(AIMessage, response)
        return AnswerDTO(
            text=response.content,
            used_tokens=response.usage_metadata.get("total_tokens", 0)
        )
    
    async def _run_agent(self, data: QuestionDTO) -> AnswerDTO:
        input_dict = {
            "input": data.text,
            "history": [(message.role, message.text) for message in data.history]
        }
        response = await self._agent.ainvoke(input_dict)
        return AnswerDTO(
            text=response["output"],
            used_tokens=0
        )