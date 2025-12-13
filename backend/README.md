# Cemantix léger (backend)

But : petit backend FastAPI en français pour un jeu de type "Cemantix" en local.

Prérequis :
- Python 3.10+
- pip

Installation :
1. Crée et active un environnement virtuel :
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .\.venv\Scripts\activate   # Windows

2. Installer les dépendances :
   pip install -r requirements.txt

Remarques :
- Par défaut la similarité est calculée avec TF-IDF (léger).
- Si tu veux activer la similarité sémantique (meilleure qualité), installe `sentence-transformers` (déjà listé dans requirements) et mets la variable d'environnement USE_ST_MODEL=1 avant de lancer.

Lancer le serveur :
   export USE_ST_MODEL=0   #Linux/macOS ou 1 pour le modèle sémantique si installé
   $env:USE_ST_MODEL = "0" #Windows ou 1 pour le modèle sémantique si installé
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Endpoints (rough) :
- POST /start    -> démarre une partie (option: fournir target pour tests)
- POST /guess    -> envoyer { game_id, guess } => réponse en français (score, rang, top_similaires)
- GET  /vocab    -> renvoie une partie du vocabulaire (utile pour debug / UI)

Le backend garde les parties en mémoire (pas persistant). Pour un usage local et test d'IA c'est suffisant.
