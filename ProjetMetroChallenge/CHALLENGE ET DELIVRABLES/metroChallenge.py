# metroChallenge.py (version mise à jour pour le challenge)
# -*- coding: utf-8 -*-
"""
Script principal pour le challenge de reconnaissance des lignes de métro.
Ce script charge un modèle pré-entraîné, traite toutes les images d'un dossier
et sauvegarde les détections dans un fichier .mat.
"""
import numpy as np
import os
import re
from PIL import Image
import scipy.io as sio
import glob

# Importer les fonctions et classes nécessaires
from myMetroProcessing import FinalMetroSystem, processOneMetroImage
from evaluationV2 import evaluation

# =============================================================================
# --- PARAMÈTRES DU CHALLENGE ---
# =============================================================================

# Répertoire contenant les images du challenge
CHALLENGE_DIRECTORY = "BD_CHALLENGE"

# Fichier de sortie pour vos résultats.
# Choisissez un nom d'équipe, par exemple 'results_Equipe4.mat'
OUTPUT_FILE = 'myChallengeResults.mat'

# Chemin vers votre meilleur modèle YOLO entraîné
MODEL_PATH = 'models/best3.pt'

# Facteur de redimensionnement utilisé.
# Si vous avez entraîné et traitez les images à leur taille originale, laissez 1.0
RESIZE_FACTOR = 1.0

# Fichier de vérité terrain pour l'évaluation locale (le correcteur utilisera le sien)
GT_FILE = 'GTCHALLENGETEST.mat'


# =============================================================================
# --- PROGRAMME PRINCIPAL ---
# =============================================================================

def run_challenge():
    """
    Exécute le processus complet du challenge.
    """
    print("🚀 DÉMARRAGE DU CHALLENGE METRO...")

    # 1. Charger le modèle YOLO pré-entraîné (une seule fois)
    # -------------------------------------------------------------------------
    print(f"🧠 Chargement du modèle depuis : '{MODEL_PATH}'")
    try:
        metro_system = FinalMetroSystem(MODEL_PATH)
        print("✅ Modèle chargé avec succès !")
    except FileNotFoundError:
        print(f"❌ ERREUR CRITIQUE : Le fichier modèle '{MODEL_PATH}' est introuvable.")
        print("   Assurez-vous que le modèle est au bon endroit et que le chemin est correct.")
        return # Arrêt du script si le modèle n'est pas trouvé

    # 2. Lire les images du répertoire du challenge
    # -------------------------------------------------------------------------
    image_paths = sorted(glob.glob(os.path.join(CHALLENGE_DIRECTORY, '*.JPG')))
    if not image_paths:
        print(f"❌ ERREUR CRITIQUE : Aucune image .JPG trouvée dans '{CHALLENGE_DIRECTORY}'.")
        return

    print(f"🖼️  {len(image_paths)} images trouvées. Début du traitement...")

    # 3. Traiter toutes les images et collecter les résultats
    # -------------------------------------------------------------------------
    all_results = []
    
    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        print(f"    -> Traitement de : {img_filename}")
        
        try:
            # Extraire le numéro de l'image (robuste à différents formats de nom)
            match = re.search(r'\((\d+)\)', img_filename)
            if not match:
                print(f"       /!\\ Attention: Impossible d'extraire le numéro de l'image pour {img_filename}. On l'ignore.")
                continue
            img_num = int(match.group(1))

            # Charger l'image au format attendu par le modèle (numpy array, 0-1)
            im_np = np.array(Image.open(img_path).convert('RGB')) / 255.0
            
            # Appeler votre fonction de traitement qui utilise le modèle YOLO
            _, bd = processOneMetroImage(img_filename, im_np, img_num, RESIZE_FACTOR, 
                                         metro_system=metro_system)
            
            # Ajouter les détections de cette image à la liste globale
            if bd.shape[0] > 0:
                all_results.append(bd)
        
        except Exception as e:
            print(f"       /!\\ ERREUR inattendue lors du traitement de {img_filename}: {e}")

    # 4. Formater et sauvegarder les résultats finaux
    # -------------------------------------------------------------------------
    if all_results:
        final_bd = np.concatenate(all_results, axis=0)
    else:
        # S'il n'y a eu aucune détection, créer un tableau vide avec la bonne structure
        final_bd = np.empty((0, 6))

    sio.savemat(OUTPUT_FILE, {'BD': final_bd})
    print(f"\n🎉 Traitement terminé ! Les résultats ont été sauvegardés dans '{OUTPUT_FILE}'.")
    
    return OUTPUT_FILE


if __name__ == "__main__":
    
    # Étape 1 : Générer le fichier de résultats
    results_file = run_challenge()
    
    # Étape 2 : Lancer l'évaluation quantitative (pour vos tests)
    # Cette partie sera exécutée par le correcteur avec le vrai GT.
    # Assurez-vous que le fichier GT_FILE existe pour que cette partie fonctionne.
    if results_file and os.path.exists(GT_FILE):
        print("\n\n=========================================================")
        print("--- DÉBUT DE L'ÉVALUATION QUANTITATIVE ---")
        print(f"   Vérité terrain: '{GT_FILE}'")
        print(f"   Vos résultats : '{results_file}'")
        print("=========================================================")
        try:
            evaluation(GT_FILE, results_file, RESIZE_FACTOR)
        except FileNotFoundError:
             print(f"\n/!\\ Le fichier de vérité terrain '{GT_FILE}' n'a pas été trouvé.")
             print("   L'évaluation quantitative est impossible sans ce fichier.")
        except Exception as e:
            print(f"\n/!\\ Une erreur est survenue pendant l'évaluation : {e}")
    else:
        print("\nSkipping evaluation: results file or GT file not found.")