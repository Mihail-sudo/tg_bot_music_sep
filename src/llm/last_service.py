import pylast


class LastFMService:
    def __init__(self, api_key):
        self.network = pylast.LastFMNetwork(api_key=api_key)

    def get_similar_artists(self, artist_name: str, limit: int = 5) -> list[dict]:
        try:
            artist = self.network.get_artist(artist_name)
            similar = artist.get_similar(limit=limit)
            return [{"name": a.item.name, "match": float(a.match) * 100} for a in similar]
        except Exception as e:
            print('error', e)
            return []
    
    def get_top_tracks_by_genre(self, genre: str, limit: int = 5) -> list[dict]:
        try:
            tag = self.network.get_tag(genre)
            top_tracks = tag.get_top_tracks(limit=limit)
            return [{"artist": t.item.artist.name, "title": t.item.title, "url": t.item.get_url()} for t in top_tracks]
        except Exception as e:
            print("Ошибка при получении треков по тегу:", e)
            return []
