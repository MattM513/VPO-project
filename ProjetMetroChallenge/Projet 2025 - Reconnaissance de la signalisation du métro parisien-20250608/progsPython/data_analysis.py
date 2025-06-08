# -*- coding: utf-8 -*-
"""
Analyse des données d'apprentissage pour comprendre les caractéristiques des signes de métro
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
import os

def analyze_training_data():
    """
    Analyse les données d'apprentissage pour extraire les caractéristiques des signes
    """
    
    # Charger les données d'apprentissage
    try:
        data = scipy.io.loadmat('Apprentissage.mat')
        BD = data['BD']
        print(f"Données chargées: {BD.shape[0]} signes d'apprentissage")
    except:
        print("Fichier Apprentissage.mat non trouvé")
        return
    
    # Analyser la distribution des lignes
    lignes = BD[:, 5].astype(int)
    lignes_uniques, counts = np.unique(lignes, return_counts=True)
    
    print("\nDistribution des lignes dans l'apprentissage:")
    for ligne, count in zip(lignes_uniques, counts):
        print(f"Ligne {ligne}: {count} occurrences")
    
    # Analyser les tailles des boîtes englobantes
    largeurs = BD[:, 2] - BD[:, 1]  # x2 - x1
    hauteurs = BD[:, 4] - BD[:, 3]  # y2 - y1
    
    print(f"\nTailles des signes:")
    print(f"Largeur moyenne: {np.mean(largeurs):.1f} ± {np.std(largeurs):.1f}")
    print(f"Hauteur moyenne: {np.mean(hauteurs):.1f} ± {np.std(hauteurs):.1f}")
    print(f"Largeur min/max: {np.min(largeurs):.0f} / {np.max(largeurs):.0f}")
    print(f"Hauteur min/max: {np.min(hauteurs):.0f} / {np.max(hauteurs):.0f}")
    
    # Visualiser quelques exemples de chaque ligne
    plt.figure(figsize=(15, 10))
    
    # Histogramme des lignes
    plt.subplot(2, 3, 1)
    plt.bar(lignes_uniques, counts)
    plt.xlabel('Numéro de ligne')
    plt.ylabel('Nombre d\'occurrences')
    plt.title('Distribution des lignes')
    
    # Histogramme des tailles
    plt.subplot(2, 3, 2)
    plt.hist(largeurs, bins=20, alpha=0.7, label='Largeurs')
    plt.hist(hauteurs, bins=20, alpha=0.7, label='Hauteurs')
    plt.xlabel('Taille (pixels)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des tailles')
    plt.legend()
    
    # Relation largeur/hauteur
    plt.subplot(2, 3, 3)
    plt.scatter(largeurs, hauteurs, alpha=0.6)
    plt.xlabel('Largeur')
    plt.ylabel('Hauteur')
    plt.title('Largeur vs Hauteur')
    
    # Analyser les couleurs dominantes pour chaque ligne
    print("\nAnalyse des couleurs par ligne (à implémenter avec les images):")
    couleurs_metro = {
        1: "Jaune", 2: "Bleu", 3: "Vert olive", 4: "Violet", 5: "Orange",
        6: "Vert clair", 7: "Rose", 8: "Bleu clair", 9: "Vert foncé", 
        10: "Marron", 11: "Marron clair", 12: "Vert", 13: "Bleu clair", 14: "Violet"
    }
    
    for ligne in lignes_uniques:
        print(f"Ligne {ligne}: {couleurs_metro.get(ligne, 'Couleur inconnue')}")
    
    plt.tight_layout()
    plt.show()
    
    return BD

def extract_sign_samples(BD, images_dir="BD", max_samples_per_line=3):
    """
    Extrait quelques échantillons de signes pour chaque ligne
    """
    if not os.path.exists(images_dir):
        print(f"Répertoire {images_dir} non trouvé")
        return
    
    lignes_uniques = np.unique(BD[:, 5].astype(int))
    
    fig, axes = plt.subplots(len(lignes_uniques), max_samples_per_line, 
                            figsize=(12, 2*len(lignes_uniques)))
    fig.suptitle('Échantillons de signes par ligne', fontsize=16)
    
    for i, ligne in enumerate(lignes_uniques):
        # Trouver les signes de cette ligne
        indices_ligne = np.where(BD[:, 5] == ligne)[0]
        
        # Prendre quelques échantillons
        echantillons = indices_ligne[:max_samples_per_line]
        
        for j, idx in enumerate(echantillons):
            try:
                # Charger l'image
                num_image = int(BD[idx, 0])
                nom_image = f"IM ({num_image}).JPG"
                chemin_image = os.path.join(images_dir, nom_image)
                
                if os.path.exists(chemin_image):
                    image = Image.open(chemin_image)
                    image_array = np.array(image)
                    
                    # Extraire la région du signe
                    x1, x2, y1, y2 = BD[idx, 1:5].astype(int)
                    signe = image_array[y1:y2, x1:x2]
                    
                    # Afficher
                    ax = axes[i, j] if len(lignes_uniques) > 1 else axes[j]
                    ax.imshow(signe)
                    ax.set_title(f'Ligne {int(ligne)}')
                    ax.axis('off')
                else:
                    ax = axes[i, j] if len(lignes_uniques) > 1 else axes[j]
                    ax.text(0.5, 0.5, 'Image\nmanquante', ha='center', va='center')
                    ax.set_title(f'Ligne {int(ligne)}')
                    ax.axis('off')
                    
            except Exception as e:
                print(f"Erreur pour la ligne {ligne}, échantillon {j}: {e}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    BD = analyze_training_data()
    if BD is not None:
        extract_sign_samples(BD)