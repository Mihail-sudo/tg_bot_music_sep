import tiktoken
import httpx
import urllib.parse
from typing import Optional


def token_counter(messages):
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += len(enc.encode(msg.content))
    return total


async def search_lyrics(artist: str, title: str, timeout: int = 10) -> Optional[str]:
    base_url = "https://api.lyrics.ovh/v1/"
    encoded_artist = urllib.parse.quote(artist)
    encoded_title = urllib.parse.quote(title)
    url = f"{base_url}{encoded_artist}/{encoded_title}"


    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            if 'lyrics' in data:
                lyrics = data['lyrics']
                return lyrics
            else:
                return "Текст песни не найден в ответе API"

    except httpx.HTTPStatusError as e:
        print(f"Ошибка HTTP: {e}")
        return None
    except httpx.RequestError as e:
        print(f"Ошибка сети: {e}")
        return None
