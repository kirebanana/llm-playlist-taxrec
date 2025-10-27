import pandas as pd

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    drop_cols = ['index', 'explicit', 'key', 'mode', 'time_signature', 'release_year']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df['artist_norm'] = df['artists'].str.lower().str.strip()
    df['track_norm'] = df['track_name'].str.lower().str.strip()

    df.dropna(inplace=True)
    df = df[df['decade'] != 0]

    return df

if __name__ == "__main__":
    df_clean = load_and_clean("../data/spotify_tracks.csv")
    print("Cleaned data:")
    print(df_clean.shape)
    print(f"Total tracks: {len(df_clean)}")


