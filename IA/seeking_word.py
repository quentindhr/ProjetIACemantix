from gensim.models import KeyedVectors

# Chemin vers le fichier que tu as téléchargé
# Exemple avec le modèle de Fauconnier (format binaire)
model_path = "chemin/vers/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"

print("Chargement du modèle (cela peut prendre un peu de temps)...")

# binary=True si c'est un .bin, binary=False si c'est un .vec (FastText texte)
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Test immédiat
word = "ordinateur"
try:
    # Trouver les mots les plus proches
    neighbors = model.most_similar(word, topn=5)
    print(f"Voisins de {word}:")
    for n_word, score in neighbors:
        print(f"- {n_word} ({score:.2f})")
        
    # Calculer le score entre deux mots (Le but du jeu Cemantix)
    score = model.similarity("ordinateur", "clavier")
    print(f"\nScore entre 'ordinateur' et 'clavier' : {score:.2f}") # Score sur 1
    
except KeyError:
    print(f"Le mot '{word}' n'est pas dans le vocabulaire.")