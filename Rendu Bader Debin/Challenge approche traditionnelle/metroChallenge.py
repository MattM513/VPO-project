# -*- coding: utf-8 -*-
"""
Détection de lignes de métro :
– Entraîne le modèle sur les données d’apprentissage
– Applique le modèle sur le jeu de challenge
– Sauvegarde les résultats et, si la vérité‑terrain est disponible, évalue les performances
"""
import os
import re
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from evaluationV2 import evaluation, compareTestandRef
from myMetroProcessing import ImprovedMetroSystem, processOneMetroImage



def draw_rectangle(x1, x2, y1, y2, color):
    """Dessine un rectangle sur le graphique courant (axes par défaut)."""
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                     linewidth=2, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)



challenge_dir = "BD_CHALLENGE"
file_out = "teamsNN.mat"
resize_factor = 1

training_images_folder = "BD_METRO"
training_gt_file = "Apprentissage.mat"


print("PHASE D'ENTRAÎNEMENT")
print("=" * 60)

if not os.path.exists(training_images_folder):
    raise FileNotFoundError(
        f"Dossier d'entraînement introuvable : {training_images_folder}"
    )

if not os.path.exists(training_gt_file):
    raise FileNotFoundError(
        f"Fichier de vérité‑terrain introuvable : {training_gt_file}"
    )

metro_system = ImprovedMetroSystem()
print("Entraînement sur les données d'apprentissage…")
metro_system.train_system(training_images_folder, training_gt_file, resize_factor)
print("Entraînement terminé")
print("=" * 60)



print("LECTURE DU RÉPERTOIRE DE CHALLENGE")

if not os.path.exists(challenge_dir):
    raise FileNotFoundError(f"Répertoire de challenge introuvable : {challenge_dir}")

image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
pattern = re.compile(r"\((\d+)\)")
numbered_images = []

for filename in os.listdir(challenge_dir):
    if filename.lower().endswith(image_extensions):
        match = pattern.search(filename)
        if match:
            numbered_images.append((int(match.group(1)), filename))

numbered_images.sort()
image_files = [filename for filename in numbered_images]

print(f"{len(image_files)} image(s) trouvée(s) dans le dossier de challenge.")


print("\nDÉTECTION EN COURS")
print("=" * 60)

BD = []
detailed_results = []

for idx, (image_number, filename) in enumerate(image_files, start=1):
    print(f"\nImage {idx}/{len(image_files)} : {filename}")

    path = os.path.join(challenge_dir, filename)
    im = np.array(Image.open(path).convert("RGB")) / 255.0

    _, bd = processOneMetroImage(
        filename,
        im,
        image_number,
        resize_factor,
        save_images=False,
        metro_system=metro_system,
    )

    if len(bd) > 0:
        lines = [int(detection[5]) for detection in bd]
        print(f"{len(bd)} détection(s) – lignes : {lines}")
        detailed_results.append(
            {
                "image": filename,
                "number": image_number,
                "detections_count": len(bd),
                "lignes": lines,
                "detections": bd.tolist(),
            }
        )
    else:
        print("Aucune détection")
        detailed_results.append(
            {
                "image": filename,
                "number": image_number,
                "detections_count": 0,
                "lignes": [],
                "detections": [],
            }
        )

    BD.extend(bd.tolist())

print("\nSAUVEGARDE DES RÉSULTATS")
print("=" * 60)

sio.savemat(file_out, {"BD": np.array(BD)})
print(f"Résultats sauvegardés dans : {file_out}")

summary_file = file_out.replace(".mat", "_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("RÉSUMÉ DES DÉTECTIONS – CHALLENGE MÉTRO\n")
    f.write("=" * 50 + "\n\n")

    total_detections = sum(r["detections_count"] for r in detailed_results)
    images_with_detections = sum(r["detections_count"] > 0 for r in detailed_results)

    f.write("STATISTIQUES GLOBALES\n")
    f.write(f"- Images traitées          : {len(detailed_results)}\n")
    f.write(f"- Images avec détection    : {images_with_detections}\n")
    f.write(f"- Total des détections     : {total_detections}\n")
    f.write(
        f"- Taux de détection        : "
        f"{images_with_detections / len(detailed_results) * 100:.1f}%\n\n"
    )

    line_counts = {}
    for result in detailed_results:
        for line in result["lignes"]:
            line_counts[line] = line_counts.get(line, 0) + 1

    f.write("RÉPARTITION PAR LIGNE\n")
    for line in sorted(line_counts):
        f.write(f"- Ligne {line} : {line_counts[line]} détection(s)\n")

    f.write("\nDÉTAIL PAR IMAGE\n")
    f.write("-" * 50 + "\n")
    for result in detailed_results:
        if result["detections_count"] > 0:
            f.write(
                f"{result['image']} (#{result['number']}) : "
                f"{result['detections_count']} détection(s) "
                f"– lignes {result['lignes']}\n"
            )
        else:
            f.write(f"{result['image']} (#{result['number']}) : Aucune détection\n")

print(f"Résumé détaillé : {summary_file}")


print("\nÉVALUATION QUANTITATIVE")
print("=" * 60)

gt_challenge_file = "GTCHALLENGETEST.mat"
try:
    if os.path.exists(gt_challenge_file):
        print("Évaluation sur les données de challenge…")
        results = evaluation(gt_challenge_file, file_out, resize_factor)
        # Afficher ici les métriques retournées par `evaluation`
    else:
        print(
            f"Avertissement : fichier de vérité‑terrain absent : "
            f"{gt_challenge_file}. Évaluation ignorée."
        )
except Exception as exc:
    print(f"Erreur lors de l'évaluation : {exc}")


print("\nTRAITEMENT TERMINÉ")
print("=" * 60)

print("FICHIERS GÉNÉRÉS")
print(f"- {file_out}")
print(f"- {summary_file}")

print("\nSTATISTIQUES FINALES")
print(f"- Images traitées        : {len(image_files)}")
print(f"- Detections totales     : {len(BD)}")
images_with_det = [r for r in detailed_results if r["detections_count"] > 0]
print(f"- Images avec détection  : {len(images_with_det)}")

if BD:
    detected_lines = [int(d[5]) for d in BD]
    print(f"- Lignes détectées       : {sorted(set(detected_lines))}")


# -----------------------------------------------------------------------------
# Utilitaire : affichage des résultats
# -----------------------------------------------------------------------------
def display_detection_summary():
    """Affiche un résumé visuel des détections."""
    if not detailed_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogramme : détections par image
    detection_counts = [r["detections_count"] for r in detailed_results]
    ax1.bar(range(len(detailed_results)), detection_counts, alpha=0.7)
    ax1.set_xlabel("Images (ordre de traitement)")
    ax1.set_ylabel("Nombre de détections")
    ax1.set_title("Détections par image")
    ax1.grid(True, alpha=0.3)

    # Histogramme : répartition par ligne
    if BD:
        detected_lines = [int(d[5]) for d in BD]
        unique_lines = sorted(set(detected_lines))
        counts = [detected_lines.count(l) for l in unique_lines]

        ax2.bar([f"L{l}" for l in unique_lines], counts, alpha=0.7)
        ax2.set_xlabel("Lignes de métro")
        ax2.set_ylabel("Nombre de détections")
        ax2.set_title("Répartition par ligne")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("challenge_statistics.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Graphiques sauvegardés dans challenge_statistics.png")


if detailed_results:
    display_detection_summary()
