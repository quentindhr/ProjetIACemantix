# Configuration des modèles LLM pour résoudre le Cemantix

Ce document explique comment utiliser un vrai modèle d'IA (LLM) au lieu des heuristiques pour résoudre le Cemantix.

## Options disponibles

### 1. Ollama (Recommandé - Local et gratuit)

**Installation :**
```bash
# Installer Ollama depuis https://ollama.ai
# Puis télécharger un modèle :
ollama pull llama3.2
# ou
ollama pull mistral
# ou
ollama pull qwen2.5
```

**Configuration :**
```bash
# Optionnel : définir l'URL si Ollama n'est pas sur localhost:11434
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"  # ou mistral, qwen2.5, etc.
```

**Utilisation :**
- Modifier le frontend pour envoyer `use_llm: true` et `llm_model: "ollama"`
- Ou appeler directement l'API : `POST /ai/solve` avec `{"game_id": "...", "use_llm": true, "llm_model": "ollama"}`

### 2. OpenAI API (Payant mais très performant)

**Installation :**
```bash
pip install openai
```

**Configuration :**
```bash
export OPENAI_API_KEY="sk-..."
```

**Utilisation :**
- Envoyer `use_llm: true` et `llm_model: "openai"`
- Le modèle par défaut est `gpt-4o-mini` (peut être changé dans le code)

### 3. Hugging Face Transformers (Local, nécessite GPU recommandé)

**Installation :**
```bash
pip install transformers torch
```

**Configuration :**
```bash
export HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"  # ou autre modèle
```

**Utilisation :**
- Envoyer `use_llm: true` et `llm_model: "huggingface"`
- ⚠️ Nécessite beaucoup de RAM/VRAM (8GB+ recommandé)

## Avantages des LLM vs Heuristiques

### Heuristiques (actuel)
- ✅ Rapide
- ✅ Pas de dépendances externes
- ❌ Stratégie fixe, peu flexible
- ❌ Ne "comprend" pas vraiment le contexte

### LLM
- ✅ Raisonne sur les indices comme un humain
- ✅ Peut découvrir des patterns complexes
- ✅ Plus flexible et adaptatif
- ❌ Plus lent (surtout Hugging Face)
- ❌ Nécessite des ressources (API key ou GPU)

## Exemple d'utilisation

```python
# Dans le frontend ou via curl
POST /ai/solve
{
  "game_id": "abc-123",
  "use_llm": true,
  "llm_model": "ollama"
}
```

## Recommandation

Pour commencer, utilisez **Ollama** avec **llama3.2** :
- Gratuit
- Local (pas de clé API)
- Bonne performance
- Facile à installer

