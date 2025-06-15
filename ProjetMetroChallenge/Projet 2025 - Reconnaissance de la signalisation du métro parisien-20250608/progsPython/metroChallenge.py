# -*- coding: utf-8 -*-
"""
Script de challenge amélioré avec système entraîné
Entraîne le modèle sur les données d'apprentissage puis l'applique au challenge
"""
import numpy as np
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.io as sio
from evaluationV2 import evaluation, compareTestandRef

# Importer le système amélioré
from myMetroProcessing import ImprovedMetroSystem, processOneMetroImage

# Fonction utilitaire pour dessiner des rectangles
def draw_rectangle(x1, x2, y1, y2, color):
    """Dessine un rectangle sur le plot actuel"""
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                    linewidth=2, edgecolor=color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)

# CONFIGURATION ================================================================
challengeDirectory = "../BD_CHALLENGE"
file_out = 'teamsNN.mat'  # Vos résultats
resize_factor = 1

# Dossiers pour l'entraînement
training_images_folder = '../BD_METRO'  # Ajustez selon votre structure
training_gt_file = 'Apprentissage.mat'

# ÉTAPE 1: ENTRAÎNEMENT DU SYSTÈME ============================================
print("🎓 PHASE D'ENTRAÎNEMENT")
print("="*60)

# Vérifier que les fichiers d'entraînement existent
if not os.path.exists(training_images_folder):
    print(f"❌ ERREUR: Dossier d'entraînement '{training_images_folder}' introuvable")
    print("   Ajustez le chemin 'training_images_folder' dans le script")
    exit(1)

if not os.path.exists(training_gt_file):
    print(f"❌ ERREUR: Fichier de vérité terrain '{training_gt_file}' introuvable")
    print("   Copiez le fichier dans le répertoire courant ou ajustez le chemin")
    exit(1)

# Créer et entraîner le système
print("Initialisation du système de détection...")
metro_system = ImprovedMetroSystem()

print("Entraînement sur les données d'apprentissage...")
metro_system.train_system(training_images_folder, training_gt_file, resize_factor)

print("✅ ENTRAÎNEMENT TERMINÉ")
print("="*60)

# ÉTAPE 2: LECTURE DU RÉPERTOIRE DE CHALLENGE =================================
print("🔍 LECTURE DU RÉPERTOIRE DE CHALLENGE")

if not os.path.exists(challengeDirectory):
    print(f"❌ ERREUR: Répertoire de challenge '{challengeDirectory}' introuvable")
    exit(1)

image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
pattern = re.compile(r'\((\d+)\)')
numbered_images = []

for filename in os.listdir(challengeDirectory):
    if filename.lower().endswith(image_extensions):
        match = pattern.search(filename)
        if match:
            number = int(match.group(1))
            numbered_images.append((number, filename))

numbered_images.sort()
imageFilesList = [filename for filename in numbered_images]

print(f"📁 {len(imageFilesList)} images trouvées dans le challenge")

# ÉTAPE 3: TRAITEMENT DE TOUTES LES IMAGES ====================================
print("\n🚀 PHASE DE DÉTECTION SUR LE CHALLENGE")
print("="*60)

num_images = len(imageFilesList)
BD = []
detailed_results = []  # Pour stocker les résultats détaillés

for n_val in range(num_images):
    
    # CHARGER L'IMAGE
    nom = imageFilesList[n_val][1]
    image_number = imageFilesList[n_val][0]
    print(f"\n📷 Image {n_val+1}/{num_images}: {nom}")
    
    im_path = os.path.join(challengeDirectory, nom)
    im = np.array(Image.open(im_path).convert('RGB')) / 255.0
    
    # TRAITER L'IMAGE avec le système entraîné
    im_resized, bd = processOneMetroImage(nom, im, image_number, resize_factor, 
                                        save_images=False, metro_system=metro_system)
    
    # Stocker les résultats détaillés
    if len(bd) > 0:
        lignes_detectees = [int(detection[5]) for detection in bd]
        detailed_results.append({
            'image': nom,
            'number': image_number,
            'detections_count': len(bd),
            'lignes': lignes_detectees,
            'detections': bd.tolist()
        })
        print(f"   ✅ {len(bd)} détection(s): Lignes {lignes_detectees}")
    else:
        detailed_results.append({
            'image': nom,
            'number': image_number,
            'detections_count': 0,
            'lignes': [],
            'detections': []
        })
        print(f"   ⚠️ Aucune détection")
    
    # AJOUTER AUX RÉSULTATS GLOBAUX
    BD.extend(bd.tolist())

# ÉTAPE 4: SAUVEGARDE DES RÉSULTATS ===========================================
print(f"\n💾 SAUVEGARDE DES RÉSULTATS")
print("="*60)

# Sauvegarder au format .mat
sio.savemat(file_out, {'BD': np.array(BD)})
print(f"✅ Résultats sauvegardés dans '{file_out}'")

# Sauvegarder un résumé détaillé
summary_file = file_out.replace('.mat', '_summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("RÉSUMÉ DES DÉTECTIONS - CHALLENGE MÉTRO\n")
    f.write("="*50 + "\n\n")
    
    total_detections = sum(len(r['detections']) for r in detailed_results)
    images_with_detections = sum(1 for r in detailed_results if r['detections_count'] > 0)
    
    f.write(f"STATISTIQUES GLOBALES:\n")
    f.write(f"- Images traitées: {len(detailed_results)}\n")
    f.write(f"- Images avec détections: {images_with_detections}\n")
    f.write(f"- Total détections: {total_detections}\n")
    f.write(f"- Taux de détection: {images_with_detections/len(detailed_results)*100:.1f}%\n\n")
    
    # Compter les lignes détectées
    lignes_counts = {}
    for result in detailed_results:
        for ligne in result['lignes']:
            lignes_counts[ligne] = lignes_counts.get(ligne, 0) + 1
    
    f.write("RÉPARTITION PAR LIGNE:\n")
    for ligne in sorted(lignes_counts.keys()):
        f.write(f"- Ligne {ligne}: {lignes_counts[ligne]} détection(s)\n")
    
    f.write(f"\nDÉTAIL PAR IMAGE:\n")
    f.write("-" * 50 + "\n")
    
    for result in detailed_results:
        f.write(f"{result['image']} (#{result['number']}): ")
        if result['detections_count'] > 0:
            f.write(f"{result['detections_count']} détection(s) - Lignes: {result['lignes']}\n")
        else:
            f.write("Aucune détection\n")

print(f"📊 Résumé détaillé sauvegardé dans '{summary_file}'")

# ÉTAPE 5: ÉVALUATION QUANTITATIVE ============================================
print(f"\n📈 ÉVALUATION QUANTITATIVE")
print("="*60)

try:
    # Vérifier si le fichier de vérité terrain existe
    gt_challenge_file = 'GTCHALLENGETEST.mat'
    if os.path.exists(gt_challenge_file):
        print("Évaluation sur les données de challenge...")
        results = evaluation(gt_challenge_file, file_out, resize_factor)
        
        # Extraire et afficher les métriques principales
        print("\n🎯 RÉSULTATS DE PERFORMANCE:")
        print("="*40)
        
        # Note: Ces métriques dépendent de ce que retourne la fonction evaluation()
        # Vous devrez peut-être adapter selon le format exact de retour
        
    else:
        print(f"⚠️ Fichier de vérité terrain '{gt_challenge_file}' non trouvé")
        print("   Impossible d'évaluer quantitativement les résultats")
        
except Exception as e:
    print(f"❌ Erreur lors de l'évaluation: {e}")

# Test avec l'exemple fourni (si disponible)
try:
    if os.path.exists('teamsEX.mat'):
        print("\n📊 Comparaison avec l'exemple de référence:")
        evaluation('GTCHALLENGETEST.mat', 'teamsEX.mat', 0.5)
    else:
        print("⚠️ Fichier de référence 'teamsEX.mat' non trouvé")
except Exception as e:
    print(f"❌ Erreur lors de la comparaison avec la référence: {e}")

# ÉTAPE 6: STATISTIQUES FINALES ET CONSEILS ===================================
print(f"\n🏁 TRAITEMENT TERMINÉ")
print("="*60)

print(f"📋 FICHIERS GÉNÉRÉS:")
print(f"   • {file_out} - Résultats au format .mat")
print(f"   • {summary_file} - Résumé détaillé")

print(f"\n📊 STATISTIQUES FINALES:")
print(f"   • {len(imageFilesList)} images traitées")
print(f"   • {len(BD)} détections totales")
print(f"   • {len([r for r in detailed_results if r['detections_count'] > 0])} images avec détections")

if len(BD) > 0:
    lignes_detectees = [int(detection[5]) for detection in BD]
    lignes_uniques = set(lignes_detectees)
    print(f"   • {len(lignes_uniques)} lignes différentes détectées: {sorted(lignes_uniques)}")

print(f"\n💡 POUR AMÉLIORER LES RÉSULTATS:")
print(f"   • Analysez le fichier {summary_file} pour identifier les échecs")
print(f"   • Vérifiez les images sans détection")
print(f"   • Ajustez les paramètres de tolérance si nécessaire")
print(f"   • Enrichissez les données d'entraînement si possible")

print(f"\n🎉 CHALLENGE TERMINÉ! Bonne chance pour l'évaluation! 🚀")


# FONCTION UTILITAIRE POUR AFFICHER LES RÉSULTATS ============================
def display_detection_summary():
    """Affiche un résumé visuel des détections"""
    if not detailed_results:
        return
        
    # Créer un graphique des statistiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique 1: Nombre de détections par image
    image_numbers = [r['number'] for r in detailed_results]
    detection_counts = [r['detections_count'] for r in detailed_results]
    
    ax1.bar(range(len(image_numbers)), detection_counts, alpha=0.7)
    ax1.set_xlabel('Images (ordre de traitement)')
    ax1.set_ylabel('Nombre de détections')
    ax1.set_title('Détections par image')
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Répartition par ligne
    if BD:
        lignes_detectees = [int(detection[5]) for detection in BD]
        lignes_uniques = sorted(set(lignes_detectees))
        lignes_counts = [lignes_detectees.count(ligne) for ligne in lignes_uniques]
        
        ax2.bar([f'L{ligne}' for ligne in lignes_uniques], lignes_counts, alpha=0.7)
        ax2.set_xlabel('Lignes de métro')
        ax2.set_ylabel('Nombre de détections')
        ax2.set_title('Répartition par ligne')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('challenge_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Graphiques sauvegardés dans 'challenge_statistics.png'")

# Appeler la fonction de résumé
if 'detailed_results' in locals() and detailed_results:
    display_detection_summary()