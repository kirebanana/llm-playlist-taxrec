import os
from dotenv import load_dotenv
import ast, json
import time

import pandas as pd
from openai import OpenAI, OpenAIError
from src.taxonomy import get_taxonomy_json
from .utils import filter_tagged_df, normalize_user_tags, to_list_safe, score_tracks

load_dotenv()

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_user_prompt(user_request):
    taxonomy_json = get_taxonomy_json()
    prompt = (
        "You are a music recommendation assistant. "
        "Given the taxonomy below and a user request, "
        "return a JSON dict with keys 'genre','mood','era','tempo','theme' " 
        "where each value is a list of the best matching categories.\n\n"
        f"Taxonomy:\n{taxonomy_json}\n\n"
        f"User request: \"{user_request}\"\n\n"
        "Return format example:\n"
        "{\"genre\": [\"rock\"], \"mood\": [\"energetic\"], \"era\": [\"80s\"], "
        "\"tempo\": [\"fast\"], \"theme\": [\"party\"]}"
        "Return ONLY valid JSON. No text, no explanations, no markdown."
    )
    return prompt


def parse_user_response(response: str):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {axis: [] for axis in json.loads(get_taxonomy_json())}


def recommend_from_prompt(
    user_request,
    tagged_df,
    N = 5,
    pause = 0.5,
    verbose = True
):
    prompt = build_user_prompt(user_request)
    try:
        resp = llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0
        )
        raw_user = resp.choices[0].message.content
        if verbose:
            print("RAW LLM USER-TAGS:", raw_user, flush=True)
        user_tags = parse_user_response(raw_user)
    except OpenAIError:
        user_tags = {}
    time.sleep(pause)
    if verbose:
        norm_preview = normalize_user_tags(user_tags)
        print("Normalized user tags:", norm_preview, flush=True)
        print("DF before filtering:", tagged_df.shape, flush=True)

    df_scored = score_tracks(tagged_df, user_tags)

    if verbose:
        print("DF after scoring:", df_scored.shape, flush=True)

    return df_scored.head(N)


if __name__ == "__main__":
    from src.data_prep import load_and_clean
    from src.tagger import batch_tag_tracks

    df_clean = load_and_clean("../data/spotify_tracks.csv")

    df_sample = df_clean.head(250)

    df_tagged = batch_tag_tracks(df_sample, batch_size=50)

    playlist = recommend_from_prompt("calm study folk 2020s", df_tagged, N=15)
    print(playlist[['track_name', 'artists', 'popularity']])

