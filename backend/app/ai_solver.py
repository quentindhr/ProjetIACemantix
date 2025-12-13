"""
Module d'IA pour résoudre automatiquement le jeu Cemantix
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Le modèle spaCy est déjà chargé dans game.py
# On l'importe depuis là
from .game import nlp


class AISolver:
    """IA qui résout le jeu Cemantix en utilisant la similarité sémantique"""
    
    def __init__(self, vocab: List[str], vocab_vectors: np.ndarray):
        self.vocab = vocab
        self.vocab_vectors = vocab_vectors
        self.used_words = set()
    
    def find_best_guess(self, history: List[Dict]) -> Optional[str]:
        """
        Trouve le meilleur mot à proposer basé sur l'historique
        
        Stratégie améliorée (SANS connaître le mot cible) :
        1. Si pas d'historique : mot commun
        2. Si score > 90% : chercher les mots les plus proches (convergence agressive)
        3. Si score > 70% : chercher dans un rayon restreint autour du meilleur guess
        4. Sinon : utiliser triangulation sémantique avec les meilleurs guesses
        """
        # Filtrer les mots déjà utilisés
        available_vocab = [w for w in self.vocab if w not in self.used_words]
        if not available_vocab:
            return None
        
        vocab_to_index = {w: i for i, w in enumerate(self.vocab)}
        available_indices = [vocab_to_index[w] for w in available_vocab if w in vocab_to_index]
        
        if not available_indices:
            return None
        
        available_vectors = self.vocab_vectors[available_indices]
        
        # Si pas d'historique, commencer avec un mot commun et représentatif
        if not history:
            # Stratégie : choisir un mot qui est sémantiquement "central" 
            # (proche de beaucoup d'autres mots dans l'espace sémantique)
            # On prend quelques mots communs et on choisit celui qui est le plus "central"
            candidates = available_vocab[:min(50, len(available_vocab))]
            
            if len(candidates) > 10:
                # Calculer la similarité moyenne de chaque candidat avec tous les autres
                candidate_indices = [vocab_to_index[w] for w in candidates if w in vocab_to_index]
                if candidate_indices:
                    candidate_vectors = self.vocab_vectors[candidate_indices]
                    # Pour chaque candidat, calculer sa similarité moyenne avec les autres
                    avg_similarities = []
                    for i, vec in enumerate(candidate_vectors):
                        vec_reshaped = vec.reshape(1, -1)
                        sims = cosine_similarity(vec_reshaped, candidate_vectors)[0]
                        # Moyenne sans compter la similarité avec soi-même (qui est 1.0)
                        avg_sim = (np.sum(sims) - 1.0) / (len(sims) - 1) if len(sims) > 1 else 0
                        avg_similarities.append(avg_sim)
                    
                    # Prendre le mot le plus "central" (haute similarité moyenne)
                    best_central_idx = np.argmax(avg_similarities)
                    return candidates[best_central_idx]
            
            # Fallback simple
            return np.random.choice(candidates)
        
        # Analyser l'historique
        best_guess = max(history, key=lambda h: h.get('score', 0))
        best_score = best_guess.get('score', 0) / 100  # Convertir de % à 0-1
        best_word = best_guess.get('guess', '')
        best_rank = best_guess.get('rank', 999999)
        
        best_word_doc = nlp(best_word)
        if not best_word_doc.has_vector:
            return np.random.choice(available_vocab)
        
        best_vec = best_word_doc.vector.reshape(1, -1)
        
        # STRATÉGIE 1 : Score très élevé (>90%) - Convergence agressive
        if best_score > 0.9:
            # Chercher les mots les plus proches du meilleur guess
            similarities = cosine_similarity(best_vec, available_vectors)[0]
            # Prendre le meilleur (ou top 2 pour un peu de variété)
            top_2_indices = np.argsort(similarities)[::-1][:2]
            best_idx = top_2_indices[0] if len(top_2_indices) == 1 else np.random.choice(top_2_indices)
            return available_vocab[best_idx]
        
        # STRATÉGIE 2 : Score élevé (>70%) - Recherche ciblée
        if best_score > 0.7:
            # Chercher les mots proches, mais avec un peu plus de variété
            similarities = cosine_similarity(best_vec, available_vectors)[0]
            # Prendre parmi les top 5
            top_5_indices = np.argsort(similarities)[::-1][:5]
            best_idx = np.random.choice(top_5_indices)
            return available_vocab[best_idx]
        
        # STRATÉGIE 3 : Score moyen (>40%) - Triangulation sémantique
        if best_score > 0.4 and len(history) >= 2:
            # Utiliser les 2 meilleurs guesses pour trianguler
            top_2_guesses = sorted(history, key=lambda h: h.get('score', 0), reverse=True)[:2]
            
            if len(top_2_guesses) == 2:
                word1 = top_2_guesses[0].get('guess', '')
                word2 = top_2_guesses[1].get('guess', '')
                score1 = top_2_guesses[0].get('score', 0) / 100
                score2 = top_2_guesses[1].get('score', 0) / 100
                
                doc1 = nlp(word1)
                doc2 = nlp(word2)
                
                if doc1.has_vector and doc2.has_vector:
                    # Calculer un vecteur interpolé entre les deux meilleurs guesses
                    # Plus le score est élevé, plus on lui donne de poids
                    vec1 = doc1.vector
                    vec2 = doc2.vector
                    
                    # Poids basés sur les scores (favoriser le meilleur)
                    weight1 = score1 ** 1.5
                    weight2 = score2 ** 1.5
                    total_weight = weight1 + weight2
                    
                    if total_weight > 0:
                        interpolated_vec = ((vec1 * weight1 + vec2 * weight2) / total_weight).reshape(1, -1)
                        similarities = cosine_similarity(interpolated_vec, available_vectors)[0]
                        
                        # Prendre parmi les top 10
                        top_10_indices = np.argsort(similarities)[::-1][:10]
                        best_idx = np.random.choice(top_10_indices)
                        return available_vocab[best_idx]
        
        # STRATÉGIE 4 : Score faible ou peu d'historique - Vecteur moyen pondéré
        if len(history) >= 2:
            # Utiliser tous les guesses avec pondération par score
            direction_vector = None
            total_weight = 0
            
            for guess_data in history:
                word = guess_data.get('guess', '')
                score = guess_data.get('score', 0) / 100
                word_doc = nlp(word)
                
                if word_doc.has_vector:
                    # Poids exponentiel : les bons scores comptent beaucoup plus
                    weight = (score ** 2) if score > 0.3 else (score * 0.5)
                    if direction_vector is None:
                        direction_vector = word_doc.vector * weight
                    else:
                        direction_vector += word_doc.vector * weight
                    total_weight += weight
            
            if direction_vector is not None and total_weight > 0:
                direction_vector = (direction_vector / total_weight).reshape(1, -1)
                similarities = cosine_similarity(direction_vector, available_vectors)[0]
                
                # Prendre parmi les top 15
                top_15_indices = np.argsort(similarities)[::-1][:15]
                best_idx = np.random.choice(top_15_indices)
                return available_vocab[best_idx]
        
        # STRATÉGIE 5 : Fallback - Chercher proche du meilleur guess
        similarities = cosine_similarity(best_vec, available_vectors)[0]
        # Prendre parmi les top 20
        top_20_indices = np.argsort(similarities)[::-1][:20]
        best_idx = np.random.choice(top_20_indices)
        return available_vocab[best_idx]
    
    def solve_game(self, game_manager, game_id: str, max_iterations: int = 6) -> Dict:
        """
        Résout automatiquement une partie de Cemantix
        
        Returns:
            Dict avec le résultat : {'success': bool, 'guesses': List, 'target': str}
        """
        if game_id not in game_manager.games:
            return {'success': False, 'error': 'Partie non trouvée'}
        
        game = game_manager.games[game_id]
        
        if game.finished:
            return {'success': False, 'error': 'Partie déjà terminée'}
        
        # Réinitialiser les mots utilisés
        self.used_words = set()
        
        guesses_made = []
        
        # Faire des guesses jusqu'à trouver ou atteindre la limite
        for iteration in range(max_iterations):
            if game.finished:
                break
            
            # Récupérer l'historique actuel
            history = [{"guess": g, "score": s * 100, "rank": r} for g, s, r in game.guesses]
            
            # Trouver le meilleur guess (SANS connaître le mot cible)
            best_guess = self.find_best_guess(history)
            
            if not best_guess:
                break
            
            # Marquer comme utilisé
            self.used_words.add(best_guess)
            
            # Faire le guess
            try:
                result = game_manager.score_guess(game_id, best_guess)
                guesses_made.append({
                    'guess': best_guess,
                    'score': result.get('score', 0),
                    'rank': result.get('rank', 0)
                })
                
                # Si trouvé, arrêter
                if result.get('finished') and result.get('won'):
                    return {
                        'success': True,
                        'guesses': guesses_made,
                        'target': result.get('target'),
                        'attempts': len(guesses_made)
                    }
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # Si on arrive ici, on n'a pas trouvé
        return {
            'success': False,
            'guesses': guesses_made,
            'target': game.target if game.finished else None,
            'attempts': len(guesses_made)
        }

