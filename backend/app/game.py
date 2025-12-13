from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import random
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Chargement du modèle de langue (contient les vecteurs sémantiques)
print("Chargement du modèle spaCy...")
try:
    nlp = spacy.load("fr_core_news_md")
except OSError:
    raise RuntimeError("Le modèle 'fr_core_news_md' n'est pas trouvé. Lancez: python -m spacy download fr_core_news_md")

class Game:
    def __init__(self, target: str, max_attempts: int = 6):
        self.id = str(uuid4())
        self.target = target
        self.attempts: int = 0
        self.max_attempts = max_attempts
        self.guesses: List[Tuple[str, float, int]] = []  # (guess, score, rank) 
        self.finished: bool = False
        self.won: bool = False

class GameManager:
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.games: Dict[str, Game] = {}
        
        # 1. Prétraitement : On ne garde que les mots connus du modèle spaCy
        # pour éviter les erreurs ou les vecteurs vides (zéro)
        self.valid_vocab = []
        vectors_list = []
        
        print("Indexation du vocabulaire...")
        # nlp.pipe est plus rapide pour traiter une liste
        for doc in nlp.pipe(self.vocab):
            # On ne garde que si le mot a un vecteur valide
            if doc.has_vector and doc.vector_norm > 0:
                self.valid_vocab.append(doc.text)
                vectors_list.append(doc.vector)
        
        self.vocab = self.valid_vocab
        # Matrice numpy contenant tous les vecteurs du vocabulaire
        self.vocab_vectors = np.array(vectors_list)
        print(f"Vocabulaire chargé : {len(self.vocab)} mots vectorisés.")

    def start_game(self, target: Optional[str] = None, max_attempts: int = 6) -> Game:
        if target is None:
            target = random.choice(self.vocab)
        # Si la cible demandée n'est pas dans notre vocabulaire vectorisé, on fallback
        if target not in self.vocab:
             # On essaye de trouver le mot s'il existe quand même dans spacy
             if not nlp(target).has_vector:
                 raise ValueError(f"Le mot cible '{target}' n'est pas connu du modèle sémantique.")
        
        g = Game(target=target, max_attempts=max_attempts)
        self.games[g.id] = g
        return g

    def score_guess(self, game_id: str, guess: str) -> Dict:
        if game_id not in self.games:
            raise KeyError("Partie introuvable")
        game = self.games[game_id]
        
        if game.finished:
            return {"error": "Partie terminée", "finished": True, "won": game.won, "target": game.target}

        game.attempts += 1
        guess_norm = guess.strip()
        
        # --- Calcul du Score ---
        target_doc = nlp(game.target)
        guess_doc = nlp(guess_norm)

        # Si le mot n'a pas de vecteur (mot inconnu / faute de frappe)
        if not guess_doc.has_vector or guess_doc.vector_norm == 0:
            score = 0.0
        else:
            # score de similarité (entre 0 et 1)
            score = float(target_doc.similarity(guess_doc))

        # --- Calcul du Rang (Top 1000, etc.) ---
        # On compare le vecteur de la CIBLE avec tout le VOCABULAIRE
        # target_vec shape: (1, 300), vocab_vectors shape: (N, 300)
        
        target_vec = target_doc.vector.reshape(1, -1)
        
        # Similarité cosinus entre la cible et TOUS les mots du vocabulaire
        # Cela renvoie un tableau [0.1, 0.5, 0.9, ...]
        sims = cosine_similarity(target_vec, self.vocab_vectors)[0]
        
        # Si le mot deviné est dans le vocabulaire, on utilise sa similarité précise issue du tableau
        # pour s'assurer que le classement est cohérent
        if guess_norm in self.vocab:
            idx = self.vocab.index(guess_norm)
            guess_score_in_vocab = sims[idx]
            # On met à jour le score affiché pour qu'il corresponde exactement au classement
            score = float(guess_score_in_vocab)

        # Combien de mots ont un score supérieur à ma proposition ?
        # C'est le rang (ex: si 5 mots sont meilleurs, je suis 6ème)
        rank = int(np.sum(sims > score)) + 1

        # Mise à jour état du jeu
        game.guesses.append((guess_norm, score, rank))

        # Condition de victoire (Score très proche de 1 ou mot identique)
        if guess_norm.lower() == game.target.lower():
            score = 1.0 # Force 1.0
            rank = 1
            game.finished = True
            game.won = True
        elif game.attempts >= game.max_attempts:
            game.finished = True
            game.won = False

        # Récupérer les mots les plus proches pour info (optionnel, aide au debug)
        # top_k_idx = sims.argsort()[::-1][:10]
        # top_k = [{"word": self.vocab[i], "sim": float(sims[i])} for i in top_k_idx]

        return {
            "game_id": game.id,
            "guess": guess_norm,
            "score": round(score * 100, 2), # En pourcentage souvent plus lisible (0-100)
            "rank": rank,
            "attempts": game.attempts,
            "remaining": max(0, game.max_attempts - game.attempts),
            "finished": game.finished,
            "won": game.won,
            "target": game.target if game.finished else None,
            "history": [{"guess": g, "score": round(s * 100, 2), "rank": r} for g, s, r in game.guesses],
        }

    def get_vocab(self, limit: int = 200) -> List[str]:
        return self.vocab[:limit]