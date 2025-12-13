"""
Module d'IA utilisant un LLM (Large Language Model) pour résoudre le Cemantix
Options : OpenAI API, Ollama (local), ou Hugging Face Transformers
"""
from typing import List, Dict, Optional
import os
import json

# Option 1 : Utiliser OpenAI API (nécessite une clé API)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Option 2 : Utiliser Ollama (modèle local, gratuit)
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Option 3 : Utiliser Hugging Face Transformers (modèle local)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class LLMSolver:
    """IA qui résout le Cemantix en utilisant un LLM pour raisonner"""
    
    def __init__(self, vocab: List[str], vocab_vectors=None, model_type: str = "ollama"):
        """
        Args:
            vocab: Liste des mots du vocabulaire
            vocab_vectors: Vecteurs du vocabulaire (optionnel, pour fallback)
            model_type: "openai", "ollama", ou "huggingface"
        """
        self.vocab = vocab
        self.vocab_vectors = vocab_vectors
        self.used_words = set()
        self.model_type = model_type
        
        # Initialiser selon le type de modèle
        if model_type == "openai" and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY non définie. Utilisez 'ollama' ou définissez la clé.")
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini"  # ou "gpt-3.5-turbo" pour moins cher
            
        elif model_type == "ollama" and OLLAMA_AVAILABLE:
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            self.model_name = os.getenv("OLLAMA_MODEL", "llama3.2")  # ou "mistral", "qwen2.5"
            
        elif model_type == "huggingface" and HF_AVAILABLE:
            model_name = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
            print(f"Chargement du modèle Hugging Face: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            print("Modèle chargé!")
        else:
            raise ValueError(f"Modèle {model_type} non disponible. Installez les dépendances nécessaires.")
    
    def _call_llm(self, prompt: str) -> str:
        """Appelle le LLM avec le prompt"""
        if self.model_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Tu es un expert en résolution de jeux de mots sémantiques. Tu dois analyser les indices et proposer le meilleur mot suivant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        
        elif self.model_type == "ollama":
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                }
            )
            return response.json()["response"].strip()
        
        elif self.model_type == "huggingface":
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + 100,
                temperature=0.7,
                do_sample=True
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extraire seulement la partie générée (après le prompt)
            return generated_text[len(prompt):].strip()
        
        return ""
    
    def _build_prompt(self, history: List[Dict], available_words: List[str]) -> str:
        """Construit le prompt pour le LLM"""
        prompt = """Tu joues à Cemantix, un jeu où tu dois trouver un mot secret en français.
Tu as proposé des mots et reçu des scores de similarité sémantique (0-100%).

Historique de tes tentatives :
"""
        for i, guess in enumerate(history, 1):
            prompt += f"{i}. Mot: '{guess['guess']}' - Score: {guess['score']:.1f}% - Rang: {guess.get('rank', 'N/A')}\n"
        
        prompt += f"""
Analyse ces indices et propose le meilleur mot suivant parmi ces options :
{', '.join(available_words[:50])}  # Limiter à 50 mots pour ne pas surcharger le prompt

Réponds UNIQUEMENT avec le mot que tu proposes, sans explication."""
        
        return prompt
    
    def find_best_guess(self, history: List[Dict]) -> Optional[str]:
        """Trouve le meilleur mot en utilisant le LLM"""
        available_vocab = [w for w in self.vocab if w not in self.used_words]
        
        if not available_vocab:
            return None
        
        # Si pas d'historique, choisir un mot commun
        if not history:
            return available_vocab[0] if available_vocab else None
        
        # Construire le prompt
        prompt = self._build_prompt(history, available_vocab)
        
        try:
            # Appeler le LLM
            response = self._call_llm(prompt)
            
            # Nettoyer la réponse (enlever guillemets, espaces, etc.)
            guess = response.strip().strip('"').strip("'").strip()
            
            # Vérifier que le mot est dans le vocabulaire disponible
            if guess.lower() in [w.lower() for w in available_vocab]:
                # Trouver la version exacte (avec la bonne casse)
                for word in available_vocab:
                    if word.lower() == guess.lower():
                        return word
            
            # Si le LLM a proposé un mot hors vocabulaire, prendre le meilleur guess basé sur l'historique
            # comme fallback
            best_guess = max(history, key=lambda h: h.get('score', 0))
            best_word = best_guess.get('guess', '')
            
            # Chercher un mot proche sémantiquement (fallback heuristique)
            from .game import nlp
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            best_doc = nlp(best_word)
            if best_doc.has_vector:
                best_vec = best_doc.vector.reshape(1, -1)
                # Utiliser les vecteurs du vocabulaire (doit être passé en paramètre)
                # Pour l'instant, fallback simple
                pass
            
        except Exception as e:
            print(f"Erreur lors de l'appel LLM: {e}")
            # Fallback : retourner le premier mot disponible
            return available_vocab[0] if available_vocab else None
        
        return available_vocab[0] if available_vocab else None
    
    def solve_game(self, game_manager, game_id: str, max_iterations: int = 6) -> Dict:
        """Résout automatiquement une partie en utilisant le LLM"""
        if game_id not in game_manager.games:
            return {'success': False, 'error': 'Partie non trouvée'}
        
        game = game_manager.games[game_id]
        
        if game.finished:
            return {'success': False, 'error': 'Partie déjà terminée'}
        
        self.used_words = set()
        guesses_made = []
        
        for iteration in range(max_iterations):
            if game.finished:
                break
            
            history = [{"guess": g, "score": s * 100, "rank": r} for g, s, r in game.guesses]
            
            best_guess = self.find_best_guess(history)
            
            if not best_guess:
                break
            
            self.used_words.add(best_guess)
            
            try:
                result = game_manager.score_guess(game_id, best_guess)
                guesses_made.append({
                    'guess': best_guess,
                    'score': result.get('score', 0),
                    'rank': result.get('rank', 0)
                })
                
                if result.get('finished') and result.get('won'):
                    return {
                        'success': True,
                        'guesses': guesses_made,
                        'target': result.get('target'),
                        'attempts': len(guesses_made)
                    }
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {
            'success': False,
            'guesses': guesses_made,
            'target': game.target if game.finished else None,
            'attempts': len(guesses_made)
        }

