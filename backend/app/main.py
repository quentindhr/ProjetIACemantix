from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from pathlib import Path

from .game import GameManager

BASE_DIR = Path(__file__).resolve().parent
VOCAB_FILE = BASE_DIR / "vocab.txt"

if not VOCAB_FILE.exists():
    raise RuntimeError(f"Fichier vocab introuvable : {VOCAB_FILE}")

with VOCAB_FILE.open(encoding="utf-8") as f:
    vocab = [line.strip() for line in f if line.strip()]

game_manager = GameManager(vocab=vocab)

app = FastAPI(title="Cemantix léger (FR)")

# Autoriser le frontend local (Angular) à accéder à l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartPayload(BaseModel):
    target: Optional[str] = None
    max_attempts: Optional[int] = 6

class GuessPayload(BaseModel):
    game_id: str
    guess: str

@app.post("/start")
def start_game(p: StartPayload):
    if p.target and p.target not in vocab:
        raise HTTPException(status_code=400, detail="Le mot cible doit appartenir au vocabulaire (ou laissez vide).")
    g = game_manager.start_game(target=p.target, max_attempts=p.max_attempts or 6)
    return {"message": "Partie démarrée", "game_id": g.id, "max_attempts": g.max_attempts}

@app.post("/guess")
def make_guess(p: GuessPayload):
    try:
        res = game_manager.score_guess(p.game_id, p.guess)
        return res
    except KeyError:
        raise HTTPException(status_code=404, detail="Partie non trouvée")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocab")
def get_vocab(limit: Optional[int] = 200):
    return {"vocab": game_manager.get_vocab(limit)}
