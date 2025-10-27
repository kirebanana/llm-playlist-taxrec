import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import score_tracks, score_tracks_intersection
from src.data_prep import load_and_clean
from src.tagger import batch_tag_tracks
from src.recommend import recommend_from_prompt

K = 10
MIN_PLAYLIST_SIZE = 5
CORPUS_FILEPATH = "data/spotify_tracks.csv"
GROUND_TRUTH_FILEPATH = "data/spotify_playlists.csv"
TAGGED_CORPUS_CACHE = "data/validation_corpus_tagged.csv"
AUDIO_FEATURE_COLS = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

def calculate_precision_recall(recommended, actual, k):
    if not recommended or not actual:
        return 0.0, 0.0

    recommended_set = set(recommended)
    actual_set = set(actual)
    intersection = len(recommended_set.intersection(actual_set))

    precision = intersection / k if k > 0 else 0.0
    recall = intersection / len(actual_set) if len(actual_set) > 0 else 0.0

    return precision, recall


def calculate_ndcg(recommended, actual, k):
    if not recommended or not actual:
        return 0.0

    relevance = np.array([1 if track in actual else 0 for track in recommended])
    true_relevance = np.zeros_like(relevance)
    true_relevance[:len(actual)] = 1

    return ndcg_score([relevance], [true_relevance], k=k)


def calculate_audio_feature_distance(recommended_ids, actual_ids, corpus_df):
    recommended_features = corpus_df[corpus_df['track_id'].isin(recommended_ids)][AUDIO_FEATURE_COLS]
    actual_features = corpus_df[corpus_df['track_id'].isin(actual_ids)][AUDIO_FEATURE_COLS]

    if recommended_features.empty or actual_features.empty:
        return 1.0  # Max distance if no features found

    recommended_vector = recommended_features.mean().values.reshape(1, -1)
    actual_vector = actual_features.mean().values.reshape(1, -1)

    similarity = cosine_similarity(recommended_vector, actual_vector)[0][0]
    distance = 1 - similarity

    return distance



if __name__ == "__main__":
    print("--- Starting Playlist Recommender Evaluation ---")

    print("Loading and cleaning data...")
    # print(f"Loading corpus from '{CORPUS_FILEPATH}'...")
    corpus_df = load_and_clean(CORPUS_FILEPATH)

    # print(f"Loading ground truth playlists from '{GROUND_TRUTH_FILEPATH}'...")
    playlists_df = pd.read_csv(
        GROUND_TRUTH_FILEPATH,
        quotechar='"',
        escapechar='\\',
        on_bad_lines='skip',
        engine='python'
    )
    # print("Ground Truth Columns:", playlists_df.columns)
    playlists_df.columns = playlists_df.columns.str.strip().str.replace('"', '')

    corpus_df['match_key'] = corpus_df['artists'].str.lower() + '|' + corpus_df['track_name'].str.lower()
    playlists_df['match_key'] = playlists_df['artistname'].str.lower() + '|' + playlists_df['trackname'].str.lower()

    print("Finding overlapping tracks...")
    validation_tracks_df = pd.merge(
        playlists_df,
        corpus_df[['track_id', 'match_key', *AUDIO_FEATURE_COLS]],
        on='match_key',
        how='inner'
    )

    ground_truth = validation_tracks_df.groupby('playlistname')['track_id'].apply(list).to_dict()

    SAMPLE_SIZE = 5000
    ground_truth = dict(list(ground_truth.items())[:SAMPLE_SIZE])
    print(f"Runnin gon a sample of {len(ground_truth)} playlists...")

    all_validation_track_ids = set([tid for t_list in ground_truth.values() for tid in t_list])
    df_to_tag = corpus_df[corpus_df['track_id'].isin(all_validation_track_ids)].drop_duplicates(subset=['track_id'])

    if os.path.exists(TAGGED_CORPUS_CACHE):
        print(f"Loading tagged validation tracks from cache: '{TAGGED_CORPUS_CACHE}'...")
        tagged_corpus_df = pd.read_csv(TAGGED_CORPUS_CACHE)
    else:
        print(f"Cache not found. Tagging {len(df_to_tag)} unique validation tracks with LLM...")
        tagged_corpus_df = batch_tag_tracks(df_to_tag, batch_size=50)
        tagged_corpus_df.to_csv(TAGGED_CORPUS_CACHE, index=False)
        print(f"Saved tagged corpus to cache: '{TAGGED_CORPUS_CACHE}'")

    print(f"\nRunning evaluation for {len(ground_truth)} playlists...")
    all_results = []

    for playlist_name, actual_track_ids in tqdm(ground_truth.items()):
        recommended_df = recommend_from_prompt(
            user_request=playlist_name,
            tagged_df=tagged_corpus_df,
            N=K,
            verbose=False
        )
        recommended_track_ids = recommended_df['track_id'].tolist()

        precision, recall = calculate_precision_recall(recommended_track_ids, actual_track_ids, K)
        ndcg = calculate_ndcg(recommended_track_ids, actual_track_ids, K)
        audio_dist = calculate_audio_feature_distance(recommended_track_ids, actual_track_ids, corpus_df)

        all_results.append({
            "playlist_name": playlist_name,
            f"precision_at_{K}": precision,
            f"recall_at_{K}": recall,
            f"ndcg_at_{K}": ndcg,
            "audio_distance": audio_dist
        })

    results_df = pd.DataFrame(all_results)

    print("\n--- Evaluation Complete ---")
    print("\nSample of Results:")
    print(results_df.head())

    print("\nAggregate Metrics:")
    print(results_df.drop(columns=['playlist_name']).mean().to_string())