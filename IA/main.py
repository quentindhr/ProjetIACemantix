"""
Script de monitoring des guess du jeu Cemantix en temps réel
Lance ce script en parallèle du backend pour voir les tentatives au fur et à mesure
"""

import requests
import time
from datetime import datetime
import seeking_word as sw

class GuessMonitor:
    def __init__(self, backend_url: str = "http://127.0.0.1:8000"):
        self.backend_url = backend_url
        self.last_guess_count = 0
        sw.load_model()  # Charger le modèle NLP une fois
        
    def check_backend(self) -> bool:
        """Vérifie que le backend est accessible"""
        try:
            response = requests.get(f"{self. backend_url}/vocab", params={"limit": 1}, timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def display_guess(self, guess:  str, score: float, rank:  int):
        """Affiche un guess dans la console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Guess: {guess} | Score: {score}% | Rang: {rank}")
    
    def get_game_history(self, game_id: str) -> dict:
        """Récupère l'historique d'une partie sans envoyer de guess"""
        try:
            # On utilise un endpoint qui existe déjà
            # Mais pour éviter d'envoyer un guess, on devrait avoir un endpoint GET dédié
            # Pour l'instant, on peut faire un call qui récupère l'info via l'erreur de partie terminée
            # OU mieux : appeler directement avec un mot vide pour voir l'historique
            
            # Meilleure solution :  ajouter un endpoint GET dans le backend
            # En attendant, on peut seulement surveiller via polling de l'historique
            pass
        except Exception as e: 
            print(f"Erreur:  {e}")
            return {}
    
    def monitor(self, game_id: str, interval: float = 0.5):
        """
        Surveille et affiche les nouveaux guess d'une partie en temps réel
        """
        print("=" * 80)
        print("🎮 MONITOR DE GUESS - Cemantix")
        print("=" * 80)
        print(f"📡 Backend    : {self.backend_url}")
        print(f"🎲 Partie     : {game_id}")
        print(f"⏱️  Intervalle :  {interval}s")
        print("-" * 80)
        
        if not self.check_backend():
            print("❌ Impossible de se connecter au backend!")
            return
        
        print("✅ Connecté au backend")
        print("👀 En attente des guess.. .\n")
        
        while True:
            try:
                # Pour récupérer l'historique sans polluer, il faut un endpoint GET
                # Le backend devrait avoir un endpoint comme GET /game/{game_id}
                response = requests.get(f"{self.backend_url}/game/{game_id}", timeout=2)
                
                if response.status_code == 200:
                    data = response.json()
                    history = data.get("history", [])
                    
                    # Afficher seulement les nouveaux guess
                    if len(history) > self.last_guess_count:
                        for i in range(self.last_guess_count, len(history)):
                            guess_data = history[i]
                            self.display_guess(
                                guess_data["guess"],
                                guess_data["score"],
                                guess_data. get("rank", "N/A")
                            )
                        self.last_guess_count = len(history)
                    
                    # Vérifier si la partie est finie
                    if data.get("finished"):
                        print("\n" + "=" * 80)
                        print("🏁 Partie terminée!")
                        if data.get("won"):
                            print(f"🎉 Le mot était: {data.get('target')}")
                        else: 
                            print(f"😢 Le mot était: {data.get('target')}")
                        print("=" * 80)
                        break
                
                time.sleep(interval)
                
            except requests.exceptions.RequestException: 
                # L'endpoint n'existe pas encore, continuer de polling
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\n\n⛔ Monitoring arrêté")
                break

def main():
    print("\n")
    
    backend_url = input("URL du backend [http://127.0.0.1:8000]: ").strip() or "http://127.0.0.1:8000"
    game_id = input("ID de la partie à surveiller:  ").strip()
    
    if not game_id:
        print("❌ Un ID de partie est requis!")
        return
    
    try:
        interval = float(input("Intervalle de polling en secondes [0.5]: ").strip() or "0.5")
    except ValueError:
        interval = 0.5
    
    print()
    
    monitor = GuessMonitor(backend_url=backend_url)
    monitor.monitor(game_id=game_id, interval=interval)

if __name__ == "__main__":
    main()