import os
from dotenv import load_dotenv
import json
import time
from typing import List, Dict
import re
import pandas as pd
from src.taxonomy import get_taxonomy_json
from openai import OpenAI, OpenAIError

load_dotenv()

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_tag_prompt(batch: List[Dict]):
    taxonomy_json = get_taxonomy_json()
    prompt_lines = [
        "You are a nuanced music classifier.",
        "For each song below, assign tags. For 'genre', 'mood', and 'theme', provide a list of 1-3 relevant tags. For 'era' and 'tempo', provide the single best tag.",
        "Use ONLY tags from this taxonomy:",
        taxonomy_json,
        "",
        "Format: [{\"track_id\": ..., \"genre\": [...], \"mood\": [...], \"era\": [...], \"tempo\": [...], \"theme\": [...]}]",
        "",
        "Songs:"
    ]

    for track in batch:
        prompt_lines.append(
            f"- track_id: {track['track_id']}, "
            f"title: '{track['track_name']}', "
            f"artist: {track['artists']}, "
            f"genre: {track.get('track_genre', 'unknown')}, "
            f"decade: {track['decade']}, "
            f"danceability: {track.get('danceability', 'N/A')}, "
            f"energy: {track.get('energy', 'N/A')}, "
            f"valence: {track.get('valence', 'N/A')}, "
            f"tempo: {track.get('tempo', 'N/A')}, "
            f"speechiness: {track.get('speechiness', 'N/A')}, "
            f"acousticness: {track.get('acousticness', 'N/A')}, "
            f"instrumentalness: {track.get('instrumentalness', 'N/A')}"
        )
    prompt_lines.append(
        "Return ONLY the raw JSON array. Do not include explanations or markdown code blocks."
    )

    return "\n".join(prompt_lines)


def parse_tag_response(response) -> List[Dict]:
    if not response or not response.strip():
        return []

    match = re.search(r'\[.*\]', response, re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("DEBUG - Found JSON-like text but failed to parse:\n", json_str)
            return []
    else:
        print("DEBUG - Could not find a JSON array in the response:\n", response)
        return []


def batch_tag_tracks(
    df,
    batch_size = 25,
    pause  = 0.5
) -> pd.DataFrame:

    records = []
    cols = [
        'track_id', 'artists', 'track_name', 'track_genre', 'decade',
        'danceability', 'energy', 'valence', 'tempo',
        'speechiness', 'acousticness', 'instrumentalness'
    ]

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i+batch_size][cols].to_dict(orient='records')
        prompt = build_tag_prompt(batch)
        try:
            resp = llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o",
            )

            tagged = parse_tag_response(resp.choices[0].message.content)

            valid_tagged = [t for t in tagged if "track_id" in t]
            if len(valid_tagged) < len(tagged):
                print(f"Warning: Dropped {len(tagged) - len(valid_tagged)} invalid records in batch starting at {i}")

            if not valid_tagged:
                print(f"Warning: Entire batch starting at {i} had no valid track_id. Skipping.")
            else:
                records.extend(valid_tagged)

        except OpenAIError as e:
            print(f"Error tagging batch starting at {i}: {e}")

        time.sleep(pause)

    tags_df = pd.DataFrame(records)

    required_axes = ['genre', 'mood', 'era', 'tempo', 'theme']
    for ax in required_axes:
        if ax not in tags_df.columns:
            tags_df[ax] = [[] for _ in range(len(tags_df))]

    def ensure_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                parsed = json.loads(x)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                return [v.strip() for v in x.split(',') if v.strip()]
        return [x]

    for ax in required_axes:
        tags_df[ax] = tags_df[ax].apply(ensure_list)

    if tags_df.empty:
        print("No tags were generated at all — returning original DataFrame with no changes.")
        return df.copy()

    return tags_df.merge(
        df[['track_id', 'popularity', 'track_name', 'artists']],
        on='track_id',
        how='left'
    )


if __name__ == "__main__":
    from src.data_prep import load_and_clean
    df_clean = load_and_clean("data/spotify_tracks.csv")

    df_tagged = batch_tag_tracks(df_clean)

    df_tagged.to_csv("data/spotify_tracks_tagged.csv", index=False)
    print("Tagging complete. Saved to data/spotify_tracks_tagged.csv")
