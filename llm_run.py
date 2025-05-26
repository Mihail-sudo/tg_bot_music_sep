import asyncio
from src.llm.llm import OllamaLLMService, QuestionDTO


async def main():
    llm = OllamaLLMService(model_name="llama3.2:3b", ollama_base_url="http://localhost:11434")
    result = await llm.execute(QuestionDTO(text="Hi, how are you?", history=[]))
    print(result)


if __name__ == "__main__":
    asyncio.run(main())