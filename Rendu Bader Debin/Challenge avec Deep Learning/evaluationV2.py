# -*- coding: utf-8 -*-
"""
Évaluation des résultats de détection.
Compare les fichiers BD de vérité terrain et de test.
"""

import numpy as np
import scipy.io

def evaluation(chemin_ref, chemin_test, facteur_redim):
    try:
        BD_REF = scipy.io.loadmat(chemin_ref)['BD']
        BD_TEST = scipy.io.loadmat(chemin_test)['BD']
    except FileNotFoundError as e:
        print(f"Erreur chargement fichiers : {e}")
        return

    BD_REF[:, 1:5] = facteur_redim * BD_REF[:, 1:5]

    I_ref = np.mean(BD_REF[:, 1:3], axis=1)
    J_ref = np.mean(BD_REF[:, 3:5], axis=1)
    D_ref = np.round((BD_REF[:, 2] - BD_REF[:, 1] + BD_REF[:, 4] - BD_REF[:, 3]) / 2)

    nb_classes = 14
    tol_dist = 0.1

    confusion = np.zeros((nb_classes, nb_classes), dtype=int)
    faux_positifs = np.zeros(nb_classes, dtype=int)
    faux_negatifs = np.zeros(nb_classes, dtype=int)
    ref_utilisee = np.zeros(BD_REF.shape[0], dtype=bool)

    for k in range(BD_TEST.shape[0]):
        symbole = BD_TEST[k, :]
        num_img = symbole[0]
        classe_pred = int(symbole[5]) - 1

        indices_ref = np.where(BD_REF[:, 0] == num_img)[0]
        if len(indices_ref) == 0:
            faux_positifs[classe_pred] += 1
            continue

        i_centre = np.mean(symbole[1:3])
        j_centre = np.mean(symbole[3:5])
        dist = np.sqrt((I_ref[indices_ref] - i_centre)**2 + (J_ref[indices_ref] - j_centre)**2)

        dmin = np.min(dist)
        idx_local = np.argmin(dist)
        idx_global = indices_ref[idx_local]

        if dmin <= tol_dist * D_ref[idx_global] and not ref_utilisee[idx_global]:
            classe_ref = int(BD_REF[idx_global, 5]) - 1
            confusion[classe_ref, classe_pred] += 1
            ref_utilisee[idx_global] = True
        else:
            faux_positifs[classe_pred] += 1

    indices_non_associes = np.where(~ref_utilisee)[0]
    for idx in indices_non_associes:
        classe = int(BD_REF[idx, 5]) - 1
        faux_negatifs[classe] += 1

    TP = np.sum(confusion)
    FP = np.sum(faux_positifs)
    FN = np.sum(faux_negatifs)

    recall = TP / (TP + FN) if TP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    acc = np.trace(confusion) / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    print('--------------------------------------------------------')
    print('SIGN DETECTION')
    print('--------------------------------------------------------')
    print(f'\t recall    = {recall:.3f}')
    print(f'\t precision = {precision:.3f}')
    print(f'\t F1-score  = {f1:.3f}')
    
    print('--------------------------------------------------------')
    print('GLOBAL AND ENLARGED ACCURACY')
    print('--------------------------------------------------------')
    print(f'\nGlobal enlarged accuracy = {acc:.3f}\n')

    print('--------------------------------------------------------')
    print('CLASS EVALUATION REPORT')
    print('--------------------------------------------------------')
    print('Ligne\t\tPrecision\tRecall\t\tf1-score\tSupport')

    precisions, recalls, f1s, supports = [], [], [], []
    
    for i in range(nb_classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp + faux_positifs[i]
        fn = np.sum(confusion[i, :]) - tp + faux_negatifs[i]
        support = tp + fn

        if support > 0:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        else:
            prec = rec = f1c = 0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1c)
        supports.append(support)

        print(f'  {i+1:2d}\t\t  {prec:.3f}\t\t {rec:.3f}\t\t  {f1c:.3f}\t\t {support:3d}')
    
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    total_support = np.sum(supports)
    weighted_prec = np.average(precisions, weights=supports)
    weighted_rec = np.average(recalls, weights=supports)
    weighted_f1 = np.average(f1s, weights=supports)

    print('--------------------------------------------------------')   
    print(f'Macro \t\t  {macro_prec:.3f}\t\t {macro_rec:.3f}\t\t  {macro_f1:.3f}\t\t {nb_classes:2d} classes')
    print(f'Weighted \t  {weighted_prec:.3f}\t\t {weighted_rec:.3f}\t\t  {weighted_f1:.3f}\t\t {total_support:3d} signs')

    for i in range(nb_classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp + faux_positifs[i]
        fn = np.sum(confusion[i, :]) - tp + faux_negatifs[i]
        support = tp + fn

        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(f'{i+1:2d}\t{prec:.3f}\t\t{rec:.3f}\t\t{f1c:.3f}\t\t{support:3d}')
