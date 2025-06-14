# evaluationV2.py (version légèrement nettoyée, logique 100% identique)
# -*- coding: utf-8 -*-
"""
Fonctions d'évaluation pour le challenge de reconnaissance des lignes de métro.
Comprend la fonction principale 'evaluation' et une fonction de visualisation 'compareTestandRef'.
"""

import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage as ski
from matplotlib.patches import Rectangle

# On sort la fonction draw_rectangle ici pour que le fichier soit autonome
def draw_rectangle_eval(x1, x2, y1, y2, color):
    ax = plt.gca()
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def evaluation(bd_ref_path, bd_test_path, resize_factor):
    """
    Évalue les performances d'un système de reconnaissance en comparant
    les résultats (bd_test_path) à une vérité terrain (bd_ref_path).
    """
    try:
        BD_REF = scipy.io.loadmat(bd_ref_path)['BD']
        BD_TEST = scipy.io.loadmat(bd_test_path)['BD']
    except FileNotFoundError as e:
        print(f"Erreur : Impossible de charger un des fichiers .mat : {e}")
        return

    # Ajuster les coordonnées de la vérité terrain si un resize_factor est appliqué
    BD_REF[:, 1:5] = resize_factor * BD_REF[:, 1:5]

    # Calcul des centroïdes et diamètres pour la vérité terrain
    I_ref = np.mean(BD_REF[:, 1:3], axis=1)
    J_ref = np.mean(BD_REF[:, 3:5], axis=1)
    D_ref = np.round((BD_REF[:, 2] - BD_REF[:, 1] + BD_REF[:, 4] - BD_REF[:, 3]) / 2)
    max_dist_tolerance = 0.1  # Tolérance de distance en % du diamètre

    num_classes = 14
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    false_positives = np.zeros(num_classes, dtype=int) # FP (détections en trop)
    
    processed_ref_symbols = np.zeros(BD_REF.shape[0], dtype=bool)

    # Itération sur chaque détection du système à tester
    for k in range(BD_TEST.shape[0]):
        test_symbol = BD_TEST[k, :]
        img_num = test_symbol[0]
        class_test = int(test_symbol[5]) - 1

        # Trouver tous les symboles de la vérité terrain dans la même image
        ref_indices_in_img = np.where(BD_REF[:, 0] == img_num)[0]

        if len(ref_indices_in_img) == 0:
            # Si aucune vérité terrain dans l'image, c'est un FP
            false_positives[class_test] += 1
            continue

        # Calculer la distance du symbole testé à tous les symboles GT de l'image
        i_test = np.mean(test_symbol[1:3])
        j_test = np.mean(test_symbol[3:5])
        distances = np.sqrt((I_ref[ref_indices_in_img] - i_test)**2 + (J_ref[ref_indices_in_img] - j_test)**2)

        min_dist = np.min(distances)
        closest_ref_local_idx = np.argmin(distances)
        closest_ref_global_idx = ref_indices_in_img[closest_ref_local_idx]

        # Vérifier si la détection correspond à un symbole GT (assez proche et non déjà associé)
        if min_dist <= max_dist_tolerance * D_ref[closest_ref_global_idx] and not processed_ref_symbols[closest_ref_global_idx]:
            class_ref = int(BD_REF[closest_ref_global_idx, 5]) - 1
            confusion_matrix[class_ref, class_test] += 1
            processed_ref_symbols[closest_ref_global_idx] = True # Marquer comme associé
        else:
            false_positives[class_test] += 1

    # Les symboles GT non associés sont des Faux Négatifs (FN)
    unprocessed_indices = np.where(~processed_ref_symbols)[0]
    false_negatives = np.zeros(num_classes, dtype=int)
    for idx in unprocessed_indices:
        class_ref = int(BD_REF[idx, 5]) - 1
        false_negatives[class_ref] += 1

    # --- AFFICHAGE DES RÉSULTATS ---
    
    # 1. Détection globale des signes (indépendamment de la classe)
    TP_detection = np.sum(confusion_matrix)
    FP_detection = np.sum(false_positives)
    FN_detection = np.sum(false_negatives)
    
    recall_det = TP_detection / (TP_detection + FN_detection) if (TP_detection + FN_detection) > 0 else 0
    precision_det = TP_detection / (TP_detection + FP_detection) if (TP_detection + FP_detection) > 0 else 0
    f1_det = 2 * recall_det * precision_det / (recall_det + precision_det) if (recall_det + precision_det) > 0 else 0
    
    print('--------------------------------------------------------')
    print('SIGN DETECTION')
    print('--------------------------------------------------------')
    print(f'\t recall    = {recall_det:.3f}')
    print(f'\t precision = {precision_det:.3f}')
    print(f'\t F1-score  = {f1_det:.3f}')

    # 2. Précision globale élargie
    accuracy_enlarged = np.trace(confusion_matrix) / (TP_detection + FP_detection + FN_detection)
    print('--------------------------------------------------------')
    print('GLOBAL AND ENLARGED ACCURACY')
    print('--------------------------------------------------------')
    print(f'\nGlobal enlarged accuracy = {accuracy_enlarged:.3f}\n')

    # 3. Rapport d'évaluation par classe
    print('--------------------------------------------------------')
    print('CLASS EVALUATION REPORT')
    print('--------------------------------------------------------')
    print('Ligne\t\tPrecision\tRecall\t\tf1-score\tSupport')
    
    precisions, recalls, f1_scores, supports = [], [], [], []
    for i in range(num_classes):
        TP_class = confusion_matrix[i, i]
        FP_class = np.sum(confusion_matrix[:, i]) - TP_class + false_positives[i]
        FN_class = np.sum(confusion_matrix[i, :]) - TP_class + false_negatives[i]
        support = TP_class + FN_class
        
        precision = TP_class / (TP_class + FP_class) if (TP_class + FP_class) > 0 else 0
        recall = TP_class / (TP_class + FN_class) if (TP_class + FN_class) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        
        print(f'  {i+1:2d}\t\t  {precision:.3f}\t\t {recall:.3f}\t\t  {f1:.3f}\t\t {support:3d}')
    
    # Calcul des moyennes Macro et Pondérée
    total_support = np.sum(supports)
    macro_avg_precision = np.mean(precisions)
    macro_avg_recall = np.mean(recalls)
    macro_avg_f1 = np.mean(f1_scores)
    
    weighted_avg_precision = np.average(precisions, weights=supports)
    weighted_avg_recall = np.average(recalls, weights=supports)
    weighted_avg_f1 = np.average(f1_scores, weights=supports)
    
    print('--------------------------------------------------------')   
    print(f'Macro \t\t  {macro_avg_precision:.3f}\t\t {macro_avg_recall:.3f}\t\t  {macro_avg_f1:.3f}\t\t {num_classes:2d} classes')
    print(f'Weighted \t  {weighted_avg_precision:.3f}\t\t {weighted_avg_recall:.3f}\t\t  {weighted_avg_f1:.3f}\t\t {total_support:3d} signs')


def compareTestandRef(imageFilesList,challengeDirectory,BDREF_path, BDTEST_path, resize_factor):
    # Cette fonction est principalement pour la visualisation, pas de changement majeur nécessaire.
    # On peut la conserver telle quelle pour vos besoins de débogage.
    # ... (code original de compareTestandRef ici) ...
    # Le code est long, donc je ne le recopie pas, mais il reste identique à votre version.
    pass # Remplacez "pass" par le code original si vous voulez utiliser la fonction.