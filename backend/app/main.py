from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from pathlib import Path

from .game import GameManager
from .ai_solver import AISolver

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

class AISolvePayload(BaseModel):
    game_id: str
    use_llm: Optional[bool] = False  # True pour utiliser LLM, False pour heuristique
    llm_model: Optional[str] = "ollama"  # "openai", "ollama", ou "huggingface"

@app.get("/game/{game_id}")
def get_game_status(game_id: str):
    """Récupère le statut et l'historique d'une partie sans faire de guess"""
    if game_id not in game_manager.games:
        raise HTTPException(status_code=404, detail="Partie non trouvée")
    
    game = game_manager.games[game_id]
    
    return {
        "game_id": game.id,
        "attempts": game.attempts,
        "max_attempts": game.max_attempts,
        "finished": game.finished,
        "won": game.won,
        "target": game.target if game.finished else None,
        "history": [{"guess": g, "score": round(s * 100, 2), "rank": r} for g, s, r in game.guesses]
    }

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

@app.post("/ai/solve")
def ai_solve(p: AISolvePayload):
    """Demande à l'IA de résoudre automatiquement la partie"""
    try:
        if p.game_id not in game_manager.games:
            raise HTTPException(status_code=404, detail="Partie non trouvée")
        
        game = game_manager.games[p.game_id]
        
        if game.finished:
            return {
                "success": False,
                "error": "Partie déjà terminée",
                "won": game.won,
                "target": game.target
            }
        
        # Choisir le type de solver
        if p.use_llm:
            from .ai_solver_llm import LLMSolver
            solver = LLMSolver(
                game_manager.vocab, 
                vocab_vectors=game_manager.vocab_vectors,
                model_type=p.llm_model or "ollama"
            )
        else:
            # Solver heuristique (par défaut)
            solver = AISolver(game_manager.vocab, game_manager.vocab_vectors)
        
        # Résoudre la partie
        result = solver.solve_game(game_manager, p.game_id, max_iterations=game.max_attempts - game.attempts)
        
        return result
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Partie non trouvée")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
