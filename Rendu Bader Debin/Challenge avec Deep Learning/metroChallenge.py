# metroChallenge.py
# -*- coding: utf-8 -*-
"""
Script principal pour le challenge de reconnaissance des lignes de métro.
Ce script charge un modèle, traite les images et sauvegarde les détections dans un fichier .mat.
"""
import numpy as np
import os
import re
from PIL import Image
import scipy.io as sio
import glob

from myMetroProcessing import FinalMetroSystem, processOneMetroImage
from evaluationV2 import evaluation

# Paramètres
CHALLENGE_DIRECTORY = "BD_CHALLENGE"
OUTPUT_FILE = 'myChallengeResults.mat'
MODEL_PATH = 'models/best4.pt'
RESIZE_FACTOR = 1.0
GT_FILE = 'GTCHALLENGETEST.mat'

def run_challenge():
    print("Début du traitement...")

    # Chargement du modèle
    try:
        metro_system = FinalMetroSystem(MODEL_PATH)
    except FileNotFoundError:
        print(f"Erreur : modèle '{MODEL_PATH}' introuvable.")
        return

    # Récupération des images
    image_paths = sorted(glob.glob(os.path.join(CHALLENGE_DIRECTORY, '*.JPG')))
    if not image_paths:
        print(f"Erreur : aucune image trouvée dans '{CHALLENGE_DIRECTORY}'.")
        return

    print(f"{len(image_paths)} images trouvées.")

    all_results = []
    
    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        print(f"Traitement de : {img_filename}")
        
        try:
            match = re.search(r'\((\d+)\)', img_filename)
            if not match:
                print(f"Nom d'image incorrect : {img_filename}")
                continue
            img_num = int(match.group(1))
            im_np = np.array(Image.open(img_path).convert('RGB')) / 255.0
            
            _, bd = processOneMetroImage(img_filename, im_np, img_num, RESIZE_FACTOR, 
                                         metro_system=metro_system)
            
            if bd.shape[0] > 0:
                all_results.append(bd)
        
        except Exception as e:
            print(f"Erreur lors du traitement de {img_filename} : {e}")

    if all_results:
        final_bd = np.concatenate(all_results, axis=0)
    else:
        final_bd = np.empty((0, 6))
    
    try:
        import pandas as pd
        df = pd.DataFrame(final_bd, columns=['image_num', 'y1', 'y2', 'x1', 'x2', 'predicted_class'])
        df.to_csv('myChallengeResults_readable.csv', index=False)
    except Exception:
        pass

    sio.savemat(OUTPUT_FILE, {'BD': final_bd})
    print(f"Résultats sauvegardés dans '{OUTPUT_FILE}'.")

    return OUTPUT_FILE


if __name__ == "__main__":
    results_file = run_challenge()

    if results_file and os.path.exists(GT_FILE):
        print("Évaluation quantitative...")
        try:
            evaluation(GT_FILE, results_file, RESIZE_FACTOR)
        except Exception as e:
            print(f"Erreur lors de l'évaluation : {e}")
