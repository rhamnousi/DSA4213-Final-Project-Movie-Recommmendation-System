#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import ast
import re
from collections import Counter
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from tqdm import tqdm

# Load and clean movies dataset
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
movies = movies[movies['id'].str.isdigit()]
movies['id'] = movies['id'].astype(int)

movies = movies[['id', 'title', 'overview', 'tagline', 'genres',
                 'release_date', 'runtime', 'vote_average', 'vote_count', 'popularity']]
movies = movies[movies['title'].notna() & movies['release_date'].notna()]
movies['overview'] = movies['overview'].fillna("")
movies['tagline'] = movies['tagline'].fillna("")

def clean_numeric(col):
    col = col.apply(lambda x: x[0] if isinstance(x, list) else x)
    return pd.to_numeric(col, errors='coerce')

for num_col in ['runtime', 'vote_average', 'vote_count', 'popularity']:
    movies[num_col] = clean_numeric(movies[num_col])
    movies[num_col] = movies[num_col].fillna(movies[num_col].median())

# Parse genres
def parse_genres(x):
    try:
        genres_list = ast.literal_eval(x) if pd.notna(x) else []
        return [g['name'] for g in genres_list if 'name' in g]
    except:
        return []

movies['genres'] = movies['genres'].apply(parse_genres)
movies['genres'] = movies['genres'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

# Merge credits
credits = pd.read_csv('credits.csv')

def safe_parse(x):
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except:
        return []

credits['cast'] = credits['cast'].apply(safe_parse)
credits['crew'] = credits['crew'].apply(safe_parse)

def get_director(crew):
    for member in crew:
        if member.get('job') == 'Director':
            return member.get('name', "")
    return ""

credits['director'] = credits['crew'].apply(get_director)
credits['top_cast'] = credits['cast'].apply(lambda c: [m.get('name') for m in c[:5]] if c else [])

# Merge keywords
keywords = pd.read_csv('keywords.csv')
keywords['keywords'] = keywords['keywords'].apply(safe_parse)
keywords['keywords_str'] = keywords['keywords'].apply(
    lambda x: " ".join([kw['name'] for kw in x if 'name' in kw])
)

# Merge all datasets
movies_merged = movies.merge(
    credits[['id', 'director', 'top_cast']], on='id', how='left'
).merge(
    keywords[['id', 'keywords_str']], on='id', how='left'
)

movies_merged['top_cast_list'] = movies_merged['top_cast'].apply(lambda x: x if isinstance(x, list) else [])
movies_merged['top_cast_str'] = movies_merged['top_cast_list'].apply(lambda xs: ", ".join(xs))

movies_merged['text'] = (
    movies_merged['title'] + " " + movies_merged['overview'] + " " + movies_merged['tagline'] + " " +
    movies_merged['genres'] + " " + movies_merged['keywords_str'].fillna("") + " " +
    movies_merged['director'].fillna("") + " " + movies_merged['top_cast_str']
).str.lower().str.replace(r'\s+', ' ', regex=True)

# Load MovieLens ratings
ratings_ml = pd.read_csv("ratings.csv")
movie_stats = ratings_ml.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('userId', 'count')
).reset_index()
movie_stats['relevance'] = movie_stats['avg_rating'] * np.log1p(movie_stats['num_ratings'])

movies_feedback = movies_merged.merge(
    movie_stats, left_on='id', right_on='movieId', how='left'
)
movies_feedback['relevance'] = movies_feedback['relevance'].fillna(0)

# Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
movie_embeddings = model.encode(movies_feedback['text'].tolist(), show_progress_bar=True)

# Feature engineering
current_year = 2025
movies_feedback['release_year'] = pd.to_datetime(movies_feedback['release_date'], errors='coerce').dt.year
movies_feedback['recency_score'] = (movies_feedback['release_year'].fillna(2000) - 1900) / (current_year - 1900)

runtime_min, runtime_max = movies_feedback['runtime'].min(), movies_feedback['runtime'].max()
movies_feedback['runtime_norm'] = (movies_feedback['runtime'] - runtime_min) / (runtime_max - runtime_min)

dir_counts = movies_feedback['director'].value_counts()
movies_feedback['director_popularity'] = movies_feedback['director'].map(dir_counts)
movies_feedback['director_popularity'] /= movies_feedback['director_popularity'].max()

all_cast = [actor for sublist in movies_feedback['top_cast_list'] for actor in sublist]
cast_counts = Counter(all_cast)
movies_feedback['cast_popularity'] = movies_feedback['top_cast_list'].apply(
    lambda x: np.mean([cast_counts[a] for a in x]) if x else 0
)
movies_feedback['cast_popularity'] /= movies_feedback['cast_popularity'].max()

movies_feedback['genre_score'] = movies_feedback['genres'].apply(lambda g: len(g.split()) / 5)
movies_feedback['overview_sentiment'] = movies_feedback['overview'].fillna("").apply(
    lambda x: (TextBlob(x).sentiment.polarity + 1) / 2
)

# Synthetic training data
KNOWN_GENRES = [
    "action", "comedy", "romance", "thriller", "horror",
    "sci-fi", "drama", "fantasy", "family", "animation", "adventure"
]

example_queries = [
    "romantic comedy", "sci-fi space adventure", "horror 1980s",
    "animated family movie", "action thriller with explosions",
    "short romantic drama", "fantasy magic adventure", "dark crime thriller"
]

train_rows = []
print("Generating synthetic training data...")
for query in tqdm(example_queries):
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, movie_embeddings)[0]
    genres_in_q = [g for g in KNOWN_GENRES if g in query.lower()]
    for i, movie in movies_feedback.iterrows():
        genre_overlap = len(set(movie['genres'].lower().split()) & set(genres_in_q)) / max(len(genres_in_q), 1)
        features = [
            sims[i], movie['vote_average'], movie['popularity'], genre_overlap,
            movie['recency_score'], movie['runtime_norm'], movie['cast_popularity'],
            movie['director_popularity'], movie['genre_score'], movie['overview_sentiment']
        ]
        label = 1 if (sims[i] > 0.35 or genre_overlap > 0.1) else 0
        train_rows.append((query, features, label))

train_df = pd.DataFrame(train_rows, columns=['query', 'features', 'label'])
X = np.vstack(train_df['features'].values)
y = train_df['label'].values
print(f"Training samples: {len(X)}, Positive labels: {y.sum()}")

xgb_model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
)
xgb_model.fit(X, y)

FEAT_COLS = [
    'similarity', 'vote_average', 'popularity', 'genre_overlap', 'recency_score',
    'runtime_norm', 'cast_popularity', 'director_popularity', 'genre_score', 'overview_sentiment'
]

# Helper functions
STOPWORDS = set("""
a an and the of with in on at for to from by into about over after before during
movie film films short long new old set around near during between
""".split())

GENRE_ALIASES = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "science-fiction": "science fiction",
    "rom-com": "romance comedy",
    "romcom": "romance comedy",
}

def normalize_genre_token(tok):
    tok = tok.lower()
    return GENRE_ALIASES.get(tok, tok)

def extract_decade_or_years(q):
    ql = q.lower()
    m = re.search(r'\b(19|20)\d0s\b', ql)
    if m:
        decade = int(ql[m.start():m.start()+4])
        return decade, decade + 9
    m = re.search(r'\b(\d{2})s\b', ql)
    if m:
        yy = int(m.group(1))
        base = 1900 if yy >= 30 else 2000
        decade = base + (yy // 10) * 10
        return decade, decade + 9
    m = re.search(r'\b(19|20)\d{2}\b', ql)
    if m:
        year = int(m.group(0))
        return year, year
    return None, None

def extract_runtime_hint(q):
    ql = q.lower()
    if "short" in ql:
        return 95
    if "under 90" in ql or "less than 90" in ql:
        return 90
    if "under 100" in ql or "less than 100" in ql:
        return 100
    return None

def build_name_vocab(df):
    names = set()
    for xs in df['top_cast_list']:
        for nm in xs:
            if nm: names.add(nm.strip())
    for nm in df['director'].fillna(""):
        if nm: names.add(nm.strip())
    return names, {n.lower(): n for n in names}

ALL_NAMES, LOWER2NAME = build_name_vocab(movies_feedback)

def extract_people(q):
    ql = q.lower()
    hits = []
    for low, orig in LOWER2NAME.items():
        if len(low) < 4:
            continue
        if low in ql:
            hits.append(orig)
    return sorted(set(hits))

def extract_genres(q):
    toks = [normalize_genre_token(t) for t in re.findall(r"[a-zA-Z\-]+", q.lower()) if t not in STOPWORDS]
    q_g = []
    for t in toks:
        if t in KNOWN_GENRES:
            q_g.append(t)
        elif t == "science" and "fiction" in toks:
            q_g.append("science fiction")
    return sorted(set(q_g))

def contains_full_name(name_list, target_name):
    tl = target_name.strip().lower()
    return any((nm or "").strip().lower() == tl for nm in name_list)

def compute_genre_overlap(movie_genres, query_genres):
    movie_set = set(movie_genres.lower().split())
    query_set = set([g.lower() for g in query_genres])
    overlap = len(movie_set & query_set)
    return overlap / max(len(query_set), 1)

# Recommendation function
def recommend_movies(
    query, top_n=10, genre=None, min_year=None, max_year=None,
    max_runtime=None, use_keywords=True, rerank_by_model=True, debug=False
):
    q_lower = query.lower()

    #Extract constraints
    inferred_min, inferred_max = extract_decade_or_years(query)
    min_year = min_year if min_year is not None else inferred_min
    max_year = max_year if max_year is not None else inferred_max
    if max_runtime is None:
        max_runtime = extract_runtime_hint(query)

    people = extract_people(query)
    genre_terms = extract_genres(query)  # can detect multiple genres

    #Base candidates
    candidates = movies_feedback.copy()

    #Filter by people
    if people:
        mask_person = False
        for person in people:
            m_cast = candidates['top_cast_list'].apply(lambda lst: contains_full_name(lst, person))
            m_dir = candidates['director'].fillna("").str.lower().eq(person.lower())
            mask_person = mask_person | m_cast | m_dir
        candidates = candidates[mask_person]

    #keep any movie that matches at least one queried genre
    if genre or genre_terms:
        terms = []
        if genre:
            terms.append(genre.lower())
        terms += [g.lower() for g in genre_terms]
        terms = sorted(set(terms))
        gmask = False
        for t in terms:
            gmask = gmask | candidates['genres'].str.contains(re.escape(t), case=False, na=False)
        candidates = candidates[gmask]

    #Year and runtime filters
    if min_year is not None or max_year is not None:
        years = pd.to_datetime(candidates['release_date'], errors='coerce').dt.year
        if min_year is not None:
            candidates = candidates[years >= min_year]
        if max_year is not None:
            candidates = candidates[years <= max_year]
    if max_runtime is not None:
        candidates = candidates[candidates['runtime'] <= max_runtime]

    if candidates.empty:
        return pd.DataFrame()

    # Similarity and scoring
    query_emb = model.encode([query])
    cand_embeddings = movie_embeddings[candidates.index]
    candidates = candidates.copy()
    candidates['similarity'] = cosine_similarity(query_emb, cand_embeddings)[0]

    # Handle multi-genre overlap
    q_genres_for_overlap = genre_terms if genre_terms else [g for g in KNOWN_GENRES if g in q_lower]
    candidates['genre_overlap'] = candidates['genres'].apply(
        lambda g: compute_genre_overlap(g, q_genres_for_overlap)
    )

    # Fill missing numeric features
    for f in ['recency_score','runtime_norm','cast_popularity','director_popularity',
              'genre_score','overview_sentiment','vote_average','popularity']:
        candidates[f] = candidates[f].fillna(0)

    # Boost for matching people
    if people:
        def person_boost(row):
            boost = 0.0
            for p in people:
                if contains_full_name(row['top_cast_list'], p) or str(row['director']).strip().lower() == p.lower():
                    boost = max(boost, 0.30)
            return boost
        candidates['person_boost'] = candidates.apply(person_boost, axis=1)
    else:
        candidates['person_boost'] = 0.0

    # Combine XGBoost + weighted features
    features = candidates[FEAT_COLS].to_numpy()
    if rerank_by_model:
        proba = xgb_model.predict_proba(features)[:, 1]
        candidates['score'] = np.clip(proba + 0.4*candidates['genre_overlap'] + candidates['person_boost'], 0, 1)
        sort_cols = ['score', 'similarity', 'genre_overlap', 'vote_average', 'popularity']
    else:
        candidates['score'] = candidates['similarity'] + 0.5*candidates['genre_overlap'] + candidates['person_boost']
        sort_cols = ['score', 'vote_average', 'popularity']

    candidates = candidates.sort_values(sort_cols, ascending=[False]*len(sort_cols))
    return candidates.head(top_n)[
        ['title','release_date','runtime','vote_average','vote_count','genres','director','top_cast_str']
    ]

# Example queries
test_queries = [
    "short sci-fi movie with Jude Law",
    "romantic comedy set in New York",
    "action thriller with explosions",
    "animated family movie",
    "horror movie in the 1980s"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = recommend_movies(query=query, top_n=5)
    print(results if not results.empty else "No matching movies found.")
