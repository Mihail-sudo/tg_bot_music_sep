import tiktoken


def token_counter(messages):
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += len(enc.encode(msg.content))
    return total