from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import random
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optionnel : utiliser sentence-transformers pour un meilleur score sémantique
USE_ST_MODEL = os.getenv("USE_ST_MODEL", "0") == "1"
if USE_ST_MODEL:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        USE_ST_MODEL = False

class Game:
    def __init__(self, target: str, max_attempts: int = 6):
        self.id = str(uuid4())
        self.target = target
        self.attempts: int = 0
        self.max_attempts = max_attempts
        self.guesses: List[Tuple[str, float]] = []  # (guess, score)
        self.finished: bool = False
        self.won: bool = False

class GameManager:
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.games: Dict[str, Game] = {}
        # TF-IDF vectorizer (traitement simple, léger)
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="french")
        self.vocab_vectors = None
        self._fit_tfidf()
        # modèle sémantique optionnel
        self.st_model = None
        if USE_ST_MODEL:
            try:
                self.st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except Exception:
                self.st_model = None

    def _fit_tfidf(self):
        # On considère chaque mot du vocabulaire comme un "document"
        try:
            self.vectorizer.fit(self.vocab)
            self.vocab_vectors = self.vectorizer.transform(self.vocab)
        except Exception:
            # Fallback minimal pour éviter crash si vocab contient caractères problématiques
            self.vectorizer = TfidfVectorizer(lowercase=True)
            self.vectorizer.fit(self.vocab)
            self.vocab_vectors = self.vectorizer.transform(self.vocab)

    def start_game(self, target: Optional[str] = None, max_attempts: int = 6) -> Game:
        if target is None:
            target = random.choice(self.vocab)
        g = Game(target=target, max_attempts=max_attempts)
        self.games[g.id] = g
        return g

    def _tfidf_score(self, a: str, b: str) -> float:
        vecs = self.vectorizer.transform([a, b])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return float(sim)

    def _semantic_score(self, a: str, b: str) -> float:
        if self.st_model is None:
            return self._tfidf_score(a, b)
        emb = self.st_model.encode([a, b])
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        return float(sim)

    def score_guess(self, game_id: str, guess: str) -> Dict:
        if game_id not in self.games:
            raise KeyError("Partie introuvable")
        game = self.games[game_id]
        if game.finished:
            return {"error": "Partie déjà terminée", "finished": True, "won": game.won, "target": game.target}

        game.attempts += 1
        guess_norm = guess.strip()

        # calcul du score entre la proposition et la cible
        if USE_ST_MODEL and self.st_model:
            score = self._semantic_score(guess_norm, game.target)
        else:
            score = self._tfidf_score(guess_norm, game.target)

        # calcul des similarités entre la cible et tout le vocabulaire pour obtenir un "rang"
        try:
            if USE_ST_MODEL and self.st_model:
                all_emb = self.st_model.encode(self.vocab + [game.target])
                tgt_emb = all_emb[-1]
                vocab_embs = all_emb[:-1]
                sims = cosine_similarity([tgt_emb], vocab_embs)[0]
            else:
                tgt_vec = self.vectorizer.transform([game.target])
                sims = cosine_similarity(tgt_vec, self.vocab_vectors)[0]
        except Exception:
            # fallback : comparer chacun avec TF-IDF si quelque chose casse
            sims = np.array([self._tfidf_score(w, game.target) for w in self.vocab])

        # score de la proposition si elle est dans le vocab sinon utiliser le score calculé
        if guess_norm in self.vocab:
            guess_idx = self.vocab.index(guess_norm)
            guess_score = float(sims[guess_idx])
        else:
            guess_score = float(score)

        # rang : combien d'éléments dans vocab ont une similarité >= guess_score
        sorted_sims = np.sort(sims)[::-1]
        rank = int((sorted_sims >= guess_score).sum())

        game.guesses.append((guess_norm, float(score)))

        if guess_norm.lower() == game.target.strip().lower():
            game.finished = True
            game.won = True
        elif game.attempts >= game.max_attempts:
            game.finished = True
            game.won = False

        top_k_idx = sims.argsort()[::-1][:10]
        top_k = [{"word": self.vocab[i], "sim": float(sims[i])} for i in top_k_idx]

        return {
            "game_id": game.id,
            "guess": guess_norm,
            "score": float(score),
            "rank": rank,
            "attempts": game.attempts,
            "remaining": max(0, game.max_attempts - game.attempts),
            "finished": game.finished,
            "won": game.won,
            "target": game.target if game.finished else None,
            "top_similaires": top_k,
            "history": [{"guess": g, "score": s} for g, s in game.guesses],
        }

    def get_vocab(self, limit: int = 200) -> List[str]:
        return self.vocab[:limit]
