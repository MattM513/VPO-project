# -*- coding: utf-8 -*-
"""
Système d'apprentissage qui utilise les vraies données d'entraînement
pour calibrer les couleurs et améliorer la détection
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import skimage as ski
from skimage import color, morphology, measure, feature
from skimage.transform import resize
import cv2
import scipy.io as sio
from PIL import Image
import os
from sklearn.cluster import KMeans
from collections import defaultdict


class MetroLearningSystem:
    """
    Système qui apprend à partir des données d'entraînement réelles
    """
    
    def __init__(self):
        # Couleurs théoriques (point de départ)
        self.theoretical_colors = {
            1: [1.0, 0.808, 0.0],          # #FFCE00
            2: [0.0, 0.392, 0.690],        # #0064B0  
            3: [0.624, 0.596, 0.145],      # #9F9825
            4: [0.753, 0.255, 0.569],      # #C04191
            5: [0.949, 0.557, 0.259],      # #F28E42
            6: [0.514, 0.769, 0.569],      # #83C491
            7: [0.953, 0.643, 0.729],      # #F3A4BA
            8: [0.808, 0.678, 0.824],      # #CEADD2
            9: [0.835, 0.788, 0.0],        # #D5C900
            10: [0.890, 0.702, 0.165],     # #E3B32A
            11: [0.553, 0.369, 0.165],     # #8D5E2A
            12: [0.0, 0.506, 0.310],       # #00814F
            13: [0.596, 0.831, 0.886],     # #98D4E2
            14: [0.400, 0.141, 0.514]      # #662483
        }
        
        # Couleurs apprises (sera rempli par l'apprentissage)
        self.learned_colors = {}
        self.color_variations = {}
        
        # Paramètres adaptatifs
        self.adaptive_tolerances = {}
        self.structure_patterns = {}
        
        # Statistiques d'apprentissage
        self.training_stats = defaultdict(list)
    
    def train_from_ground_truth(self, images_folder, gt_file, resize_factor=1.0):
        """
        Entraîne le système à partir des données d'apprentissage
        """
        print("🎓 DÉMARRAGE APPRENTISSAGE À PARTIR DES DONNÉES RÉELLES")
        print("="*60)
        
        # Charger la vérité terrain
        gt_data = sio.loadmat(gt_file)['BD']
        print(f"Données d'apprentissage chargées: {len(gt_data)} annotations")
        
        # Organiser par ligne de métro
        samples_by_line = defaultdict(list)
        
        # Analyser chaque annotation
        for annotation in gt_data:
            img_num, x1, x2, y1, y2, ligne = annotation
            img_num, x1, x2, y1, y2, ligne = int(img_num), int(x1), int(x2), int(y1), int(y2), int(ligne)
            
            # Charger l'image correspondante
            img_path = os.path.join(images_folder, f'IM ({img_num}).JPG')
            if os.path.exists(img_path):
                image = np.array(Image.open(img_path).convert('RGB')) / 255.0
                
                if resize_factor != 1.0:
                    image = resize(image, 
                                 (int(image.shape[0] * resize_factor), 
                                  int(image.shape[1] * resize_factor)),
                                 anti_aliasing=True, preserve_range=True).astype(image.dtype)
                    x1, x2, y1, y2 = (np.array([x1, x2, y1, y2]) * resize_factor).astype(int)
                
                # Analyser cette région
                sample_data = self._analyze_training_sample(image, x1, x2, y1, y2, ligne, img_num)
                if sample_data:
                    samples_by_line[ligne].append(sample_data)
        
        # Apprendre les couleurs pour chaque ligne
        print(f"\n📊 ANALYSE PAR LIGNE:")
        for ligne in sorted(samples_by_line.keys()):
            samples = samples_by_line[ligne]
            print(f"\nLigne {ligne}: {len(samples)} échantillons")
            self._learn_line_characteristics(ligne, samples)
        
        # Calculer les seuils adaptatifs
        self._compute_adaptive_thresholds()
        
        # Résumé de l'apprentissage
        self._print_learning_summary()
        
        return self.learned_colors, self.adaptive_tolerances
    
    def _analyze_training_sample(self, image, x1, x2, y1, y2, ligne, img_num):
        """
        Analyse un échantillon d'entraînement
        """
        # Extraire la région
        region = image[y1:y2, x1:x2]
        h, w = region.shape[:2]
        
        if h < 10 or w < 10:  # Région trop petite
            return None
        
        # Estimer le centre et le rayon
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 2
        
        # Analyser les couleurs avec plusieurs méthodes
        color_data = self._extract_multiple_colors(region, center_x, center_y, radius)
        
        # Analyser la structure (contraste, forme)
        structure_data = self._analyze_structure_features(region, center_x, center_y, radius)
        
        # Tentative de détection de chiffre
        digit_data = self._analyze_digit_features(region, center_x, center_y, radius)
        
        return {
            'ligne': ligne,
            'img_num': img_num,
            'region_size': (w, h),
            'radius': radius,
            'colors': color_data,
            'structure': structure_data,
            'digit': digit_data,
            'bbox': (x1, x2, y1, y2)
        }
    
    def _extract_multiple_colors(self, region, center_x, center_y, radius):
        """
        Extrait les couleurs avec plusieurs méthodes
        """
        h, w = region.shape[:2]
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        colors = {}
        
        # Méthode 1: Anneau externe (couleur de la ligne)
        ring_mask = (distances >= radius * 0.5) & (distances <= radius * 0.9)
        if np.any(ring_mask):
            ring_pixels = region[ring_mask]
            colors['ring_mean'] = np.mean(ring_pixels, axis=0)
            colors['ring_median'] = np.median(ring_pixels, axis=0)
            
            # Clustering pour trouver les couleurs dominantes
            if len(ring_pixels) > 10:
                kmeans = KMeans(n_clusters=min(3, len(ring_pixels)), random_state=42, n_init=10)
                labels = kmeans.fit_predict(ring_pixels)
                # Prendre la couleur du cluster le plus grand
                largest_cluster = np.bincount(labels).argmax()
                colors['ring_dominant'] = kmeans.cluster_centers_[largest_cluster]
        
        # Méthode 2: Bordure externe
        border_mask = distances >= radius * 0.8
        if np.any(border_mask):
            border_pixels = region[border_mask]
            colors['border_mean'] = np.mean(border_pixels, axis=0)
        
        # Méthode 3: Zones spécifiques (haut, bas, gauche, droite du cercle)
        for direction, (dx, dy) in [('top', (0, -0.7)), ('bottom', (0, 0.7)), 
                                   ('left', (-0.7, 0)), ('right', (0.7, 0))]:
            sample_x = int(center_x + dx * radius)
            sample_y = int(center_y + dy * radius)
            if 0 <= sample_x < w and 0 <= sample_y < h:
                # Prendre une petite zone autour du point
                x1, x2 = max(0, sample_x-3), min(w, sample_x+3)
                y1, y2 = max(0, sample_y-3), min(h, sample_y+3)
                if x2 > x1 and y2 > y1:
                    colors[f'{direction}_sample'] = np.mean(region[y1:y2, x1:x2], axis=(0,1))
        
        return colors
    
    def _analyze_structure_features(self, region, center_x, center_y, radius):
        """
        Analyse les caractéristiques structurelles
        """
        h, w = region.shape[:2]
        gray = color.rgb2gray(region)
        
        # Contraste centre/périphérie
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        center_mask = distances <= radius * 0.3
        ring_mask = (distances >= radius * 0.6) & (distances <= radius * 0.9)
        
        structure = {}
        
        if np.any(center_mask) and np.any(ring_mask):
            center_brightness = np.mean(gray[center_mask])
            ring_brightness = np.mean(gray[ring_mask])
            structure['contrast'] = center_brightness - ring_brightness
            structure['center_brightness'] = center_brightness
            structure['ring_brightness'] = ring_brightness
        
        # Circularité (utiliser les contours)
        binary = gray > np.mean(gray)
        contours = measure.find_contours(binary, 0.5)
        if contours:
            # Prendre le plus grand contour
            largest_contour = max(contours, key=len)
            structure['contour_length'] = len(largest_contour)
            
            # Calculer la circularité approximative
            if len(largest_contour) > 10:
                area = len(largest_contour)  # Approximation
                perimeter = len(largest_contour)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    structure['circularity'] = circularity
        
        # Détection de bords
        edges = feature.canny(gray, sigma=1.0)
        structure['edge_density'] = np.sum(edges) / edges.size
        
        return structure
    
    def _analyze_digit_features(self, region, center_x, center_y, radius):
        """
        Analyse pour détecter les chiffres au centre
        """
        h, w = region.shape[:2]
        gray = color.rgb2gray(region)
        
        # Zone centrale pour le chiffre
        center_size = max(5, radius // 2)
        x1 = max(0, center_x - center_size)
        x2 = min(w, center_x + center_size)
        y1 = max(0, center_y - center_size)
        y2 = min(h, center_y + center_size)
        
        center_region = gray[y1:y2, x1:x2]
        
        digit_features = {}
        
        if center_region.size > 0:
            # Binarisation adaptative
            if center_region.std() > 0.1:  # Assez de contraste
                threshold = np.mean(center_region) + 0.1
                binary_center = center_region > threshold
                
                digit_features['white_ratio'] = np.sum(binary_center) / binary_center.size
                digit_features['brightness_mean'] = np.mean(center_region)
                digit_features['brightness_std'] = np.std(center_region)
                
                # Détection de formes dans la zone centrale
                if np.any(binary_center):
                    labeled = measure.label(binary_center)
                    regions = measure.regionprops(labeled)
                    if regions:
                        # Prendre la plus grande région
                        largest_region = max(regions, key=lambda r: r.area)
                        digit_features['object_area'] = largest_region.area
                        digit_features['object_eccentricity'] = largest_region.eccentricity
        
        return digit_features
    
    def _learn_line_characteristics(self, ligne, samples):
        """
        Apprend les caractéristiques d'une ligne à partir des échantillons
        """
        if not samples:
            return
        
        # Collecter toutes les couleurs
        all_ring_colors = []
        all_contrasts = []
        all_digit_features = []
        
        for sample in samples:
            colors = sample['colors']
            if 'ring_mean' in colors:
                all_ring_colors.append(colors['ring_mean'])
            
            if 'ring_dominant' in colors:
                all_ring_colors.append(colors['ring_dominant'])
            
            structure = sample['structure']
            if 'contrast' in structure:
                all_contrasts.append(structure['contrast'])
            
            all_digit_features.append(sample['digit'])
        
        if all_ring_colors:
            # Couleur moyenne et variations
            all_ring_colors = np.array(all_ring_colors)
            mean_color = np.mean(all_ring_colors, axis=0)
            std_color = np.std(all_ring_colors, axis=0)
            
            self.learned_colors[ligne] = mean_color
            self.color_variations[ligne] = {
                'mean': mean_color,
                'std': std_color,
                'samples': all_ring_colors,
                'count': len(all_ring_colors)
            }
            
            # Distance par rapport à la couleur théorique
            theoretical = np.array(self.theoretical_colors[ligne])
            distance_to_theory = np.linalg.norm(mean_color - theoretical)
            
            print(f"   Couleur apprise: {mean_color}")
            print(f"   Couleur théorique: {theoretical}")
            print(f"   Distance théorie: {distance_to_theory:.3f}")
            print(f"   Variation (std): {std_color}")
        
        # Apprendre les caractéristiques structurelles
        if all_contrasts:
            self.structure_patterns[ligne] = {
                'contrast_mean': np.mean(all_contrasts),
                'contrast_std': np.std(all_contrasts),
                'contrast_min': np.min(all_contrasts),
                'contrast_max': np.max(all_contrasts)
            }
            print(f"   Contraste moyen: {np.mean(all_contrasts):.3f} ± {np.std(all_contrasts):.3f}")
    
    def _compute_adaptive_thresholds(self):
        """
        Calcule des seuils adaptatifs basés sur l'apprentissage
        """
        print(f"\n🎯 CALCUL DES SEUILS ADAPTATIFS:")
        
        for ligne in self.learned_colors:
            if ligne in self.color_variations:
                variations = self.color_variations[ligne]
                
                # Seuil basé sur 2 écarts-types
                color_std = np.mean(variations['std'])
                adaptive_tolerance = max(0.1, min(0.4, color_std * 3))
                
                self.adaptive_tolerances[ligne] = adaptive_tolerance
                print(f"   Ligne {ligne}: tolérance adaptative = {adaptive_tolerance:.3f}")
    
    def _print_learning_summary(self):
        """
        Affiche un résumé de l'apprentissage
        """
        print(f"\n📋 RÉSUMÉ DE L'APPRENTISSAGE:")
        print(f"   Lignes apprises: {sorted(self.learned_colors.keys())}")
        print(f"   Couleurs théoriques vs réelles:")
        
        for ligne in sorted(self.learned_colors.keys()):
            theo = self.theoretical_colors[ligne]
            learned = self.learned_colors[ligne]
            distance = np.linalg.norm(np.array(learned) - np.array(theo))
            print(f"     L{ligne}: distance = {distance:.3f}")
    
    def detect_with_learned_features(self, image, debug=False):
        """
        Détection utilisant les caractéristiques apprises
        """
        if not self.learned_colors:
            print("⚠️ Aucune couleur apprise ! Lancez d'abord train_from_ground_truth()")
            return []
        
        if debug:
            print("🔍 Détection avec caractéristiques apprises...")
        
        # Détecter les cercles (méthode existante)
        circles = self._detect_circles_permissive(image)
        
        if debug:
            print(f"   Cercles candidats: {len(circles)}")
        
        # Valider avec les caractéristiques apprises
        valid_detections = []
        for i, (x, y, r) in enumerate(circles):
            validation = self._validate_with_learned_features(image, x, y, r)
            
            if validation['is_valid']:
                detection = {
                    'bbox': validation['bbox'],
                    'center': (x, y),
                    'radius': r,
                    'ligne': validation['ligne'],
                    'confidence': validation['confidence'],
                    'method': 'learned'
                }
                valid_detections.append(detection)
                
                if debug:
                    print(f"   ✅ Cercle {i+1}: Ligne {validation['ligne']} "
                          f"(conf: {validation['confidence']:.2f})")
        
        return valid_detections
    
    def _detect_circles_permissive(self, image):
        """
        Détection de cercles avec paramètres permissifs
        """
        gray = color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        gray_uint8 = cv2.medianBlur(gray_uint8, 5)
        
        circles = cv2.HoughCircles(
            gray_uint8, cv2.HOUGH_GRADIENT,
            dp=1, minDist=40, param1=30, param2=25,
            minRadius=15, maxRadius=70
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Filtrage basique
            h, w = image.shape[:2]
            valid_circles = []
            for x, y, r in circles:
                if (20 < x < w-20 and 20 < y < h-20):
                    valid_circles.append((x, y, r))
            return valid_circles[:20]  # Limiter à 20
        
        return []
    
    def _validate_with_learned_features(self, image, x, y, r):
        """
        Validation utilisant les caractéristiques apprises
        """
        # Extraire région
        h, w = image.shape[:2]
        margin = 10
        x1, x2 = max(0, x-r-margin), min(w, x+r+margin)
        y1, y2 = max(0, y-r-margin), min(h, y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return {'is_valid': False, 'confidence': 0.0}
        
        region = image[y1:y2, x1:x2]
        
        # Extraire les couleurs de la région
        region_h, region_w = region.shape[:2]
        region_center_x, region_center_y = region_w // 2, region_h // 2
        
        colors = self._extract_multiple_colors(region, region_center_x, region_center_y, r)
        
        # Comparer avec les couleurs apprises
        best_ligne = -1
        best_confidence = 0.0
        
        for ligne, learned_color in self.learned_colors.items():
            # Tester différentes extractions de couleur
            confidences = []
            
            for color_key in ['ring_mean', 'ring_dominant', 'border_mean']:
                if color_key in colors:
                    extracted_color = colors[color_key]
                    distance = np.linalg.norm(extracted_color - learned_color)
                    
                    # Utiliser la tolérance adaptative si disponible
                    tolerance = self.adaptive_tolerances.get(ligne, 0.3)
                    confidence = max(0, 1.0 - distance / tolerance)
                    confidences.append(confidence)
            
            if confidences:
                ligne_confidence = max(confidences)  # Prendre la meilleure
                if ligne_confidence > best_confidence:
                    best_confidence = ligne_confidence
                    best_ligne = ligne
        
        # Validation finale
        is_valid = best_ligne != -1 and best_confidence > 0.5
        
        return {
            'is_valid': is_valid,
            'ligne': best_ligne,
            'confidence': best_confidence,
            'bbox': (x1, x2, y1, y2) if is_valid else None
        }


def processOneMetroImageLearned(nom, im, n, resizeFactor, learning_system, save_images=False):
    """
    Version utilisant le système d'apprentissage
    """
    # Redimensionnement
    if resizeFactor != 1:
        im_resized = resize(
            im,
            (int(im.shape[0] * resizeFactor), int(im.shape[1] * resizeFactor)),
            anti_aliasing=True,
            preserve_range=True
        ).astype(im.dtype)
    else:
        im_resized = im
    
    print(f"\n{'='*60}")
    print(f"Traitement image AVEC APPRENTISSAGE: {nom}")
    print(f"{'='*60}")
    
    # Détection avec caractéristiques apprises
    detections = learning_system.detect_with_learned_features(im_resized, debug=True)
    
    # Convertir au format attendu
    bd = []
    for detection in detections:
        if detection['bbox']:
            x1, x2, y1, y2 = detection['bbox']
            bd.append([n, x1, x2, y1, y2, detection['ligne']])
    
    bd = np.array(bd) if bd else np.empty((0, 6))
    
    # Affichage
    plt.figure(figsize=(16, 10))
    plt.imshow(im_resized)
    
    colors = ['lime', 'red', 'blue', 'orange', 'purple', 'brown', 'pink']
    
    for i, detection in enumerate(detections):
        if detection['bbox']:
            x1, x2, y1, y2 = detection['bbox']
            
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=4, 
                           edgecolor=colors[i % len(colors)], 
                           facecolor='none')
            plt.gca().add_patch(rect)
            
            plt.text(x1, y1-10, f"L{detection['ligne']}", 
                    color='white', fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=colors[i % len(colors)], alpha=0.9))
            
            plt.text(x1, y1-35, "LEARNED", 
                    color='white', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", 
                            facecolor='green', alpha=0.7))
    
    titre = f"{nom} - {len(detections)} détection(s) AVEC APPRENTISSAGE"
    if detections:
        lignes = [str(d['ligne']) for d in detections]
        confs = [f"{d['confidence']:.2f}" for d in detections]
        titre += f"\nLignes: {', '.join(lignes)} | Conf: {', '.join(confs)}"
    
    plt.title(titre, fontsize=12, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"🎯 Résultat AVEC APPRENTISSAGE: {len(detections)} signe(s) détecté(s)")
    for det in detections:
        print(f"  ✅ Ligne {det['ligne']} (confiance: {det['confidence']:.3f})")
    
    return im_resized, bd


if __name__ == "__main__":
    print("Système d'apprentissage Metro chargé.")
    print("Utilisation:")
    print("1. system = MetroLearningSystem()")
    print("2. system.train_from_ground_truth('../BD_METRO', 'Apprentissage.mat')")
    print("3. Utiliser processOneMetroImageLearned() pour la détection")

    learning_system = MetroLearningSystem()

    # Apprendre à partir des vraies données
    learning_system.train_from_ground_truth('../BD_METRO', 'Apprentissage.mat')
    processOneMetroImageLearned()