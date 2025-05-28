import tiktoken


def token_counter(messages):
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += len(enc.encode(msg.content))
    return total


def requires_tool(query: str) -> bool:
    # """Проверяет, нужно ли использовать агента и инструменты"""
    keywords = ["lyrics"]
    return any(kw in query.lower() for kw in keywords)
