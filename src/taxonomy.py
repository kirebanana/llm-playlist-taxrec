import json

TAXONOMY = {
    "genre": [
        "rock", "pop", "hip-hop", "electronic", "jazz", "classical",
        "country", "reggae", "latin", "rnb", "metal", "punk",
        "folk", "blues", "soul", "disco", "edm", "indie",
        "world", "k-pop", "afrobeat", "salsa", "techno",
        "dance", "ambient"
    ],
    "mood": [
        "energetic", "chill", "melancholic", "romantic", "upbeat",
        "dark", "happy", "relaxed"
    ],
    "era": ["pre-50s", "50s", "60s", "70s", "80s", "90s", "2000s", "2010s", "2020s"],
    "tempo": ["slow", "medium", "fast"],
    "theme": [
        "love", "party", "motivation", "nostalgia", "gaming",
        "workout", "study", "sleep"
    ]
}

def get_taxonomy_json() -> str:
    return json.dumps(TAXONOMY, indent=2)

if __name__ == "__main__":
    print("Taxonomy preview:")
    print(get_taxonomy_json())

