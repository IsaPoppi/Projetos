from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import json

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Recomendações Musicais")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 1) Carregar datasets
MUSIC_CSV = os.getenv(
    "MUSIC_CSV",
    "C:/Users/isadora.durante/Downloads/top_musicas/top50MusicFrom2010-2019.csv"
)
df = pd.read_csv(MUSIC_CSV)

# CSV de likes dos usuários (opcional; vamos também salvar nele)
USER_LIKES_CSV = os.getenv("USER_LIKES_CSV", str(BASE_DIR / "user_likes.csv"))
likes_df: Optional[pd.DataFrame] = None
if os.path.exists(USER_LIKES_CSV):
    try:
        likes_df = pd.read_csv(USER_LIKES_CSV)
        # normaliza nomes de colunas (aceita variações comuns)
        rename_likes = {
            "song": "title",
            "track": "title",
            "music": "title",
            "user": "user_id",
            "usuario": "user_id"
        }
        likes_df = likes_df.rename(columns=rename_likes)
        # valida colunas
        if not {"user_id", "title"} <= set(likes_df.columns):
            print("⚠️ user_likes.csv encontrado, mas sem colunas obrigatórias {user_id, title}. Ignorando.")
            likes_df = None
    except Exception as e:
        print(f"⚠️ Falha ao ler user_likes.csv: {e}")
        likes_df = None

# 2) Normalização de colunas do CSV de músicas
renomear = {
    "the genre of the track": "genre",
    "Beats.Per.Minute -The tempo of the song": "bpm",
    "Energy- The energy of a song - the higher the value, the more energtic": "energy",
    "Danceability - The higher the value, the easier it is to dance to this song": "danceability",
    "Loudness/dB - The higher the value, the louder the song": "loudness_db",
    "Liveness - The higher the value, the more likely the song is a live recording": "liveness",
    "Valence - The higher the value, the more positive mood for the song": "valence",
    "Length - The duration of the song": "length",
    "Acousticness - The higher the value the more acoustic the song is": "acousticness",
    "Speechiness - The higher the value the more spoken word the song contains": "speechiness",
    "Popularity- The higher the value the more popular the song is": "popularity",
}
df = df.rename(columns=renomear)

track_col = "title"
artist_col = "artist"
genre_col  = "genre"
year_col   = "year"
pop_col    = "popularity"

# Features numéricas para conteúdo
FEATURES = [
    "bpm", "energy", "danceability", "loudness_db", "liveness",
    "valence", "length", "acousticness", "speechiness", "popularity"
]

# Garantir tipos numéricos + imputação simples
for f in FEATURES:
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    else:
        print(f"⚠️ Coluna ausente no CSV: {f}")

df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))

# Normaliza (0..1) para aumentar estabilidade do cosseno
scaler = MinMaxScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# 3) Índices / matrizes
_similarity_matrix = cosine_similarity(df[FEATURES])
track_to_idx = {str(t): i for i, t in enumerate(df[track_col].astype(str).values)}

# Estado de likes em memória: {user_id: [titles]}
user_likes: Dict[str, List[str]] = {}
if likes_df is not None:
    likes_df["title"] = likes_df["title"].astype(str)
    valid_titles = set(df[track_col].astype(str))
    likes_df = likes_df[likes_df["title"].isin(valid_titles)]
    for uid, grp in likes_df.groupby("user_id"):
        ordered_unique = list(dict.fromkeys(grp["title"].tolist()))
        user_likes[str(uid)] = ordered_unique

# 4) Helpers — Conteúdo

def _content_scores(song_title: str) -> (np.ndarray, int):
    if song_title not in track_to_idx:
        raise KeyError("song_title não encontrado no dataset.")
    idx = track_to_idx[song_title]
    return _similarity_matrix[idx].copy(), idx

def _apply_weights_and_cosine(base_idx: int, weights: Dict[str, float]) -> np.ndarray:
    feat = df[FEATURES].copy()
    w_vec = np.ones(len(FEATURES), dtype=float)
    if weights:
        for i, f in enumerate(FEATURES):
            if f in weights:
                try:
                    w_vec[i] = float(weights[f])
                except:
                    pass
    if np.allclose(w_vec, 0.0):
        w_vec = np.ones_like(w_vec)
    feat_weighted = feat.values * w_vec.reshape(1, -1)
    base_vec = feat_weighted[base_idx].reshape(1, -1)
    sim = cosine_similarity(base_vec, feat_weighted)[0]
    smin, smax = float(sim.min()), float(sim.max())
    if smax > smin:
        sim = (sim - smin) / (smax - smin)
    else:
        sim = sim * 0.0
    return sim

def _top_k(scores_vec: np.ndarray, base_idx: int, k: int):
    order = np.argsort(scores_vec)[::-1]
    recs = []
    for j in order:
        if j == base_idx:
            continue
        recs.append({
            "track": str(df.iloc[j][track_col]),
            "artist": str(df.iloc[j][artist_col]),
            "genre":  str(df.iloc[j][genre_col]),
            "year":   int(df.iloc[j][year_col]),
            "popularity": float(df.iloc[j][pop_col]),
            "score": float(scores_vec[j]),
        })
        if len(recs) >= k:
            break
    return recs

# 5) Helpers — Colaborativo (coocorrência)

def _build_cooccurrence_scores(user_id: str, max_candidates: int = 5000) -> Dict[str, float]:
    if user_id not in user_likes or not user_likes[user_id]:
        return {}
    liked = set(user_likes[user_id])
    scores: Dict[str, float] = {}
    for other, liked_list in user_likes.items():
        if other == user_id or not liked_list:
            continue
        other_set = set(liked_list)
        inter = liked.intersection(other_set)
        if not inter:
            continue
        weight = len(inter)  # similaridade simples
        for m in other_set:
            if (m not in liked) and (m in track_to_idx):
                scores[m] = scores.get(m, 0.0) + weight
    if not scores:
        return {}
    mval = max(scores.values())
    if mval > 0:
        for k in list(scores.keys()):
            scores[k] = scores[k] / mval
    ordered = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_candidates])
    return ordered

# 6) Schemas

class ContentWeightsRequest(BaseModel):
    limit: int = Field(5, ge=1, le=100)
    weights: Optional[Dict[str, float]] = None

class GenreArtistRequest(BaseModel):
    genre: Optional[str] = None
    artist: Optional[str] = None
    limit: int = Field(5, ge=1, le=100)

class HybridRequest(BaseModel):
    song_title: str
    user_id: str
    content_weight: float = Field(0.7, ge=0.0, le=1.0)
    collab_weight: float = Field(0.3, ge=0.0, le=1.0)
    limit: int = Field(5, ge=1, le=100)
    content_feature_weights: Optional[Dict[str, float]] = None

# ---- novos schemas para criar usuários e editar likes ----
class CreateUserReq(BaseModel):
    user_id: str

class LikesUpdateReq(BaseModel):
    titles: List[str]
    mode: str = "merge"  # "merge" ou "replace"

# 7) Endpoints — UI helpers
@app.get("/index", response_class=HTMLResponse)
def home(request: Request):
    titles = sorted(df[track_col].dropna().astype(str).unique().tolist())
    genres = sorted(df[genre_col].dropna().astype(str).unique().tolist())
    artists = sorted(df[artist_col].dropna().astype(str).unique().tolist())
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "titles": titles, "genres": genres, "artists": artists},
    )

@app.get("/filters/users")
def list_users():
    return {"users": sorted(user_likes.keys())}

@app.get("/recommendations/collaborative/user/{user_id}")
def collaborative_user_profile(user_id: str):
    return {"user_id": user_id, "likes": user_likes.get(user_id, [])}

@app.get("/filters/artists-by-genre")
def artists_by_genre(genre: Optional[str] = Query(None, description="Gênero para filtrar artistas")):
    if genre:
        mask = df[genre_col].astype(str).str.contains(genre, case=False, na=False)
        artists = sorted(df.loc[mask, artist_col].dropna().astype(str).unique().tolist())
    else:
        artists = sorted(df[artist_col].dropna().astype(str).unique().tolist())
    return {"genre": genre, "artists": artists}

@app.get("/filters/genres-by-artist")
def genres_by_artist(artist: Optional[str] = Query(None, description="Artista para filtrar gêneros")):
    if artist:
        mask = df[artist_col].astype(str).str.strip().str.casefold() == artist.strip().casefold()
        genres = sorted(df.loc[mask, genre_col].dropna().astype(str).unique().tolist())
    else:
        genres = sorted(df[genre_col].dropna().astype(str).unique().tolist())
    return {"artist": artist, "genres": genres}

# 8) Endpoints — Conteúdo
@app.get("/recommendations/content-based/{song_title}")
async def content_based_recommendations_get(
    song_title: str,
    limit: int = 5,
    weights: Optional[str] = None
):
    try:
        _, base_idx = _content_scores(song_title)
    except KeyError:
        raise HTTPException(status_code=404, detail="Música não encontrada no dataset.")
    if weights:
        try:
            weights_dict = json.loads(weights)
        except Exception:
            raise HTTPException(status_code=400, detail="weights deve ser JSON válido.")
        base_scores = _apply_weights_and_cosine(base_idx, weights_dict)
    else:
        base_scores = _content_scores(song_title)[0]
        smin, smax = float(base_scores.min()), float(base_scores.max())
        if smax > smin:
            base_scores = (base_scores - smin) / (smax - smin)
        else:
            base_scores = base_scores * 0.0
    return {"base_song": song_title, "recommendations": _top_k(base_scores, base_idx, limit)}

@app.post("/recommendations/content-based/{song_title}")
async def content_based_recommendations_post(song_title: str, body: ContentWeightsRequest):
    try:
        _, base_idx = _content_scores(song_title)
    except KeyError:
        raise HTTPException(status_code=404, detail="Música não encontrada no dataset.")
    base_scores = _apply_weights_and_cosine(base_idx, body.weights or {})
    return {"base_song": song_title, "recommendations": _top_k(base_scores, base_idx, body.limit)}

# 9) Endpoints — Gênero / Artista (ordenado por popularidade)
@app.post("/recommendations/genre-artist")
async def genre_artist_recommendations(request: GenreArtistRequest):
    result = df.copy()
    if request.genre:
        result = result[result[genre_col].astype(str).str.contains(request.genre, case=False, na=False)]
    if request.artist:
        a = request.artist.strip().casefold()
        result = result[result[artist_col].astype(str).str.strip().str.casefold() == a]
    if result.empty:
        return {"recommendations": []}
    result = result.sort_values(by=pop_col, ascending=False).head(request.limit)
    cols = [track_col, artist_col, genre_col, year_col, pop_col]
    return {"recommendations": result[cols].to_dict(orient="records")}

# 10) Endpoints — Colaborativo (coocorrência)
@app.get("/recommendations/collaborative/{user_id}")
async def collaborative_recommendations(user_id: str, likes: Optional[str] = None, limit: int = 5):
    # se likes vier na query, mescla com o perfil do usuário
    if likes:
        titles = [s.strip() for s in likes.split(",") if s.strip()]
        valid = [t for t in titles if t in track_to_idx]
        if valid:
            prev = user_likes.get(user_id, [])
            merged = list(dict.fromkeys(prev + valid))
            user_likes[user_id] = merged

    scores = _build_cooccurrence_scores(user_id)
    if not scores:
        return {"recommendations": []}

    recs = []
    for track, sc in list(scores.items())[:limit]:
        j = track_to_idx[track]
        recs.append({
            "track": track,
            "artist": str(df.iloc[j][artist_col]),
            "genre":  str(df.iloc[j][genre_col]),
            "year":   int(df.iloc[j][year_col]),
            "popularity": float(df.iloc[j][pop_col]),
            "score": float(sc),
        })
    return {"recommendations": recs}

# 10.1) NOVOS — gerenciamento de usuários/likes e persistência
def _save_user_likes_csv():
    """Salva user_likes em CSV (colunas: user_id, title)."""
    rows = []
    for uid, lst in user_likes.items():
        for t in lst:
            rows.append({"user_id": uid, "title": t})
    pd.DataFrame(rows).to_csv(USER_LIKES_CSV, index=False)

@app.post("/users")
def create_user(req: CreateUserReq):
    uid = req.user_id.strip()
    if not uid:
        raise HTTPException(400, "user_id vazio")
    if uid in user_likes:
        return {"ok": True, "message": "Usuário já existia", "user_id": uid}
    user_likes[uid] = []
    return {"ok": True, "message": "Usuário criado", "user_id": uid}

@app.post("/users/{user_id}/likes")
def update_user_likes(user_id: str, req: LikesUpdateReq):
    uid = user_id.strip()
    if not uid:
        raise HTTPException(400, "user_id vazio")
    if uid not in user_likes:
        user_likes[uid] = []
    # normaliza títulos contra o catálogo
    catalog = set(df[track_col].astype(str))
    titles = [t for t in (s.strip() for s in req.titles) if t in catalog]
    if req.mode == "replace":
        user_likes[uid] = list(dict.fromkeys(titles))
    else:  # merge
        prev = user_likes.get(uid, [])
        user_likes[uid] = list(dict.fromkeys(prev + titles))
    return {"ok": True, "user_id": uid, "likes": user_likes[uid]}

@app.post("/users/save")
def save_users_csv():
    _save_user_likes_csv()
    return {"ok": True, "file": USER_LIKES_CSV}

# 11) Endpoints — Híbrido (conteúdo + colaborativo)
@app.post("/recommendations/hybrid")
async def hybrid_recommendations(request: HybridRequest):
    # --- 1) pesos globais: normaliza para somarem 1.0
    sw = request.content_weight + request.collab_weight
    if sw <= 0:
        raise HTTPException(status_code=400, detail="A soma dos pesos deve ser > 0")
    cw = request.content_weight / sw
    kw = request.collab_weight  / sw

    # --- 2) Conteúdo (com ou sem pesos por feature)
    try:
        _, base_idx = _content_scores(request.song_title)
    except KeyError:
        raise HTTPException(status_code=404, detail="Música não encontrada")

    if request.content_feature_weights:
        # usa a versão ponderada (já normalizada 0..1 dentro da helper)
        content_vec = _apply_weights_and_cosine(base_idx, request.content_feature_weights)
    else:
        # matriz padrão + normalização 0..1
        content_vec = _content_scores(request.song_title)[0]
        cmin, cmax = float(content_vec.min()), float(content_vec.max())
        content_vec = (content_vec - cmin) / (cmax - cmin) if cmax > cmin else content_vec * 0.0

    # --- 3) Colaborativo (co-ocorrência; pode ser todo zero)
    collab_scores = _build_cooccurrence_scores(request.user_id)
    collab_vec = np.zeros(len(df), dtype=float)
    for track, sc in collab_scores.items():
        j = track_to_idx.get(track)
        if j is not None:
            collab_vec[j] = sc  # já está em 0..1

    # --- 4) Combinação convexa
    hybrid = cw * content_vec + kw * collab_vec

    return {"recommendations": _top_k(hybrid, base_idx, request.limit)}

# 12) Populares (com filtros opcionais)
@app.get("/recommendations/popular")
async def popular_recommendations(year: Optional[int] = None, genre: Optional[str] = None, limit: int = 5):
    result = df.copy()
    if year is not None:
        result = result[result[year_col] == year]
    if genre:
        result = result[result[genre_col].astype(str).str.contains(genre, case=False, na=False)]
    if result.empty:
        return {"recommendations": []}
    result = result.sort_values(by=pop_col, ascending=False).head(limit)
    cols = [track_col, artist_col, genre_col, year_col, pop_col]
    return {"recommendations": result[cols].to_dict(orient="records")}

# 13) Main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
