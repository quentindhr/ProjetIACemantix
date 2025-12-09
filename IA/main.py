

import requests
import time
from datetime import datetime
from typing import Optional

class GuessMonitor:
    def __init__(self, backend_url: str = "http://127.0.0.1:8000"):
        self.backend_url = backend_url
        self.seen_guesses = set()
        
    def get_game_info(self, game_id: str) -> Optional[dict]:
        """Récupère les informations d'une partie"""
        try: 
            # Note: Il faudrait un endpoint dédié dans le backend pour lister les guess
            # Pour l'instant on simule avec les informations disponibles
            response = requests.get(f"{self.backend_url}/vocab", params={"limit": 10})
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion:  {e}")
            return None
    
    def monitor(self, game_id: str, interval: int = 2):
        """
        Surveille et affiche les guess d'une partie
        
        Args:
            game_id: ID de la partie à surveiller
            interval:  Intervalle en secondes entre chaque vérification
        """
        print(f"Monitoring de la partie:  {game_id}")
        print(f"Backend: {self.backend_url}")
        print(f"⏱Intervalle: {interval}s")
        print("-" * 60)
        
        while True: 
            try:
                # Ici, il faudrait un endpoint pour récupérer les guess
                # Pour l'instant, affiche un message d'info
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] En attente de nouveaux guess...")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n\nMonitoring arrêté par l'utilisateur")
                break
            except Exception as e:
                print(f"Erreur:  {e}")
                time.sleep(interval)

def main():
    """Point d'entrée principal"""
    print("=" * 60)
    print("🎮 MONITOR DE GUESS - Cemantix")
    print("=" * 60)
    print()
    
    # Configuration
    backend_url = input("URL du backend [http://127.0.0.1:8000]: ").strip()
    if not backend_url:
        backend_url = "http://127.0.0.1:8000"
    
    game_id = input("ID de la partie à surveiller:  ").strip()
    if not game_id:
        print("Un ID de partie est requis!")
        return
    
    try:
        interval = int(input("Intervalle de polling en secondes [2]: ").strip() or "2")
    except ValueError:
        interval = 2
    
    print()
    
    # Démarrer le monitoring
    monitor = GuessMonitor(backend_url=backend_url)
    monitor.monitor(game_id=game_id, interval=interval)

if __name__ == "__main__":
    main()