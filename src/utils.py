from typing import Any, Dict, List
import ast, json
import pandas as pd
from src.taxonomy import get_taxonomy_json

def to_list_safe(x):
    """Return a Python list for many common representations (list, tuple, stringified list, comma string, NaN)."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, str):
        # try JSON
        try:
            parsed = json.loads(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # try python literal
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [p.strip() for p in x.split(',') if p.strip()]
    return [x]

def normalize_user_tags(user_tags:Dict[str, List[str]]) -> Dict[str, List[str]] :
    """Normalize LLM output: lowercased axis keys and list-of-str values. Unknown axes are dropped."""
    tax_axes = list(json.loads(get_taxonomy_json()).keys())  # ['genre','mood','era','tempo','theme']
    norm_user_tags = {}
    for k, v in (user_tags or {}).items():
        k_norm = str(k).strip().lower()
        if k_norm not in tax_axes:
            continue
        if isinstance(v, str):
            vals = [v.strip()]
        elif isinstance(v, (list, tuple, set)):
            vals = [str(x).strip() for x in v if x is not None]
        else:
            try:
                vals = [str(x).strip() for x in list(v) if x is not None]
            except Exception:
                vals = []
        vals = [x for x in vals if x]
        norm_user_tags[k_norm] = vals
    for ax in tax_axes:
        norm_user_tags.setdefault(ax, [])
    return norm_user_tags

def filter_tagged_df(tagged_df, user_tags):
    """
    Minimal safe filter: normalize tags, coerce list columns, and filter.
    """
    df = tagged_df.copy()
    norm_tags = normalize_user_tags(user_tags)
    col_map = {c.lower(): c for c in df.columns}

    for axis, selected in norm_tags.items():
        if not selected:
            continue
        if axis not in col_map:
            continue
        col = col_map[axis]
        df[col] = df[col].apply(to_list_safe)
        mask = df[col].apply(lambda vals: any(s in vals for s in selected))
        if len(mask) != len(df):
            mask = mask.reindex(df.index, fill_value=False)
        df = df.loc[mask]
    return df

def score_tracks(tagged_df, user_tags):
    df = tagged_df.copy()
    norm_tags = normalize_user_tags(user_tags)

    weights = {
        'genre': 3,
        'mood': 2,
        'theme': 2,
        'era': 1,
        'tempo': 1
    }

    df['score'] = df['popularity'] / 100.0

    for axis, selected_tags in norm_tags.items():
        if not selected_tags:
            continue

        if axis in df.columns:
            df[axis] = df[axis].apply(to_list_safe)

            weight = weights.get(axis, 1)
            mask = df[axis].apply(lambda track_tags: any(s in track_tags for s in selected_tags))
            df.loc[mask, 'score'] += weight

    return df.sort_values('score', ascending=False)


def score_tracks_intersection(tagged_df: pd.DataFrame, user_tags: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Scores tracks based on the simple intersection count, as described
    in the TAXREC paper.
    """
    df = tagged_df.copy()
    norm_tags = normalize_user_tags(user_tags)

    # Combine all user tags into a single set for efficient intersection
    all_user_tags = set()
    for tags in norm_tags.values():
        all_user_tags.update(tags)

    def calculate_intersection(row):
        # Combine all of a track's tags into one list
        track_tags_list = []
        for axis in ['genre', 'mood', 'theme', 'era', 'tempo']:
            if axis in row and isinstance(row[axis], list):
                track_tags_list.extend(row[axis])

        # Return the size of the intersection
        return len(set(track_tags_list) & all_user_tags)

    # Apply the intersection calculation
    df['score'] = df.apply(calculate_intersection, axis=1)

    # Use popularity as a tie-breaker
    return df.sort_values(['score', 'popularity'], ascending=[False, False])