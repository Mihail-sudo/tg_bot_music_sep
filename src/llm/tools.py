from langchain_core.tools import tool
from pydantic import Field
from typing import Optional
import httpx
import urllib

@tool
async def search_lyrics(artist: str, title: str, timeout: int = 10) -> Optional[str]:
    """find a lyrics for song by the artist name and song name"""
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



tools = [search_lyrics]
