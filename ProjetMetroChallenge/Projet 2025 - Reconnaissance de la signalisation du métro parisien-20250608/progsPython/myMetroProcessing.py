# -*- coding: utf-8 -*-
"""
SYSTÈME AMÉLIORÉ pour la détection de signes de métro
Approche: Détection de cercles + Validation par couleurs + Apprentissage adaptatif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import cv2
from skimage import color, filters, morphology, measure
from skimage.transform import resize
import scipy.io as sio
from PIL import Image
import os
from collections import defaultdict
from sklearn.cluster import KMeans

class ImprovedMetroSystem:
    """
    Système amélioré de détection de signes de métro
    """
    
    def __init__(self):
        # Couleurs théoriques des lignes (RGB normalisé 0-1)
        self.theoretical_colors = {
            1: np.array([1.0, 0.808, 0.0]),      # #FFCE00
            2: np.array([0.0, 0.392, 0.690]),    # #0064B0  
            3: np.array([0.624, 0.596, 0.145]),  # #9F9825
            4: np.array([0.753, 0.255, 0.569]),  # #C04191
            5: np.array([0.949, 0.557, 0.259]),  # #F28E42
            6: np.array([0.514, 0.769, 0.569]),  # #83C491
            7: np.array([0.953, 0.643, 0.729]),  # #F3A4BA
            8: np.array([0.808, 0.678, 0.824]),  # #CEADD2
            9: np.array([0.835, 0.788, 0.0]),    # #D5C900
            10: np.array([0.890, 0.702, 0.165]), # #E3B32A
            11: np.array([0.553, 0.369, 0.165]), # #8D5E2A
            12: np.array([0.0, 0.506, 0.310]),   # #00814F
            13: np.array([0.596, 0.831, 0.886]), # #98D4E2
            14: np.array([0.400, 0.141, 0.514])  # #662483
        }
        
        # Couleurs apprises pendant l'entraînement
        self.learned_colors = {}
        self.color_tolerances = {}
        self.is_trained = False
    
    def train_system(self, images_folder, gt_file, resize_factor=1.0):
        """
        Entraîne le système avec les données d'apprentissage
        """
        print("🎓 ENTRAÎNEMENT DU SYSTÈME DE DÉTECTION")
        print("="*50)
        
        # Charger la vérité terrain
        gt_data = sio.loadmat(gt_file)['BD']
        print(f"Données d'apprentissage: {len(gt_data)} annotations")
        
        # Organiser les échantillons par ligne
        samples_by_line = defaultdict(list)
        
        for annotation in gt_data:
            img_num, x1_orig, x2_orig, y1_orig, y2_orig, ligne = annotation
            img_num, ligne = int(img_num), int(ligne)
            
            # CORRECTION des coordonnées inversées
            # Dans le fichier .mat: x1,x2,y1,y2 mais en réalité c'est y1,y2,x1,x2
            y1, y2 = int(x1_orig), int(x2_orig)  # x devient y
            x1, x2 = int(y1_orig), int(y2_orig)  # y devient x
            
            # Charger l'image correspondante
            img_path = os.path.join(images_folder, f'IM ({img_num}).JPG')
            if os.path.exists(img_path):
                image = np.array(Image.open(img_path).convert('RGB')) / 255.0
                
                # Redimensionner si nécessaire
                if resize_factor != 1.0:
                    image = resize(image, 
                                 (int(image.shape[0] * resize_factor), 
                                  int(image.shape[1] * resize_factor)),
                                 anti_aliasing=True, preserve_range=True)
                    x1, x2, y1, y2 = (np.array([x1, x2, y1, y2]) * resize_factor).astype(int)
                
                # Extraire la région et analyser la couleur
                region = image[y1:y2, x1:x2]
                if region.size > 0:
                    avg_color = self._extract_dominant_color(region)
                    samples_by_line[ligne].append(avg_color)
        
        # Apprendre les couleurs pour chaque ligne
        for ligne in sorted(samples_by_line.keys()):
            samples = samples_by_line[ligne]
            if samples:
                self._learn_line_color(ligne, samples)
        
        self.is_trained = True
        print(f"✅ ENTRAÎNEMENT TERMINÉ: {len(self.learned_colors)} lignes apprises")
        
    def _extract_dominant_color(self, region):
        """
        Extrait la couleur dominante d'une région (en évitant le blanc du chiffre)
        """
        h, w = region.shape[:2]
        
        # Convertir en HSV pour filtrer
        hsv_region = color.rgb2hsv(region)
        
        # Masquer les pixels très clairs (blanc du chiffre)
        value_channel = hsv_region[:, :, 2]
        saturation_channel = hsv_region[:, :, 1]
        
        # Garder les pixels colorés (saturation > 0.3 et valeur < 0.9)
        colored_mask = (saturation_channel > 0.3) & (value_channel < 0.9)
        
        if np.any(colored_mask):
            # Utiliser seulement les pixels colorés
            colored_pixels = region[colored_mask]
            return np.mean(colored_pixels, axis=0)
        else:
            # Fallback: utiliser la bordure de la région
            border_pixels = np.concatenate([
                region[0, :].reshape(-1, 3),  # top
                region[-1, :].reshape(-1, 3), # bottom
                region[:, 0].reshape(-1, 3),  # left
                region[:, -1].reshape(-1, 3)  # right
            ])
            return np.mean(border_pixels, axis=0)
    
    def _learn_line_color(self, ligne, color_samples):
        """
        Apprend la couleur moyenne et la tolérance pour une ligne - Version équilibrée
        """
        colors_array = np.array(color_samples)
        
        # Couleur moyenne
        mean_color = np.mean(colors_array, axis=0)
        
        # Calculer la variation pour ajuster la tolérance
        std_color = np.std(colors_array, axis=0)
        variation = np.mean(std_color)
        
        # Tolérance équilibrée - ni trop stricte ni trop permissive
        base_tolerance = 0.12  # Base raisonnable
        adaptive_tolerance = min(0.25, base_tolerance + variation * 2)  # Plafond à 0.25
        
        self.learned_colors[ligne] = mean_color
        self.color_tolerances[ligne] = adaptive_tolerance
        
        # Afficher les résultats d'apprentissage
        theoretical = self.theoretical_colors[ligne]
        distance = np.linalg.norm(mean_color - theoretical)
        
        print(f"  Ligne {ligne}: {len(color_samples)} échantillons")
        print(f"    Couleur apprise: {mean_color}")
        print(f"    Distance théorique: {distance:.3f}")
        print(f"    Tolérance: {adaptive_tolerance:.3f}")
    
    def detect_metro_signs(self, image, debug=True):
        """
        Détecte les signes de métro avec approche équilibrée
        """
        if debug:
            print(f"🔍 DÉTECTION AVEC SYSTÈME {'ENTRAÎNÉ' if self.is_trained else 'THÉORIQUE'}")
        
        # 1. Détection des cercles candidats avec approche équilibrée
        circles = self._detect_circles(image)
        if debug:
            print(f"   Cercles candidats: {len(circles)}")
        
        if not circles:
            return []
        
        # 2. Validation par couleur avec seuils équilibrés
        valid_detections = []
        colors_to_use = self.learned_colors if self.is_trained else self.theoretical_colors
        tolerances_to_use = self.color_tolerances if self.is_trained else {l: 0.18 for l in colors_to_use}
        
        for i, (x, y, r) in enumerate(circles):
            validation = self._validate_circle_by_color(image, x, y, r, colors_to_use, tolerances_to_use)
            
            if validation['is_valid']:
                detection = {
                    'center': (x, y),
                    'radius': r,
                    'bbox': validation['bbox'],
                    'ligne': validation['ligne'],
                    'confidence': validation['confidence'],
                    'color_distance': validation['color_distance']
                }
                valid_detections.append(detection)
                
                if debug:
                    print(f"   ✅ Cercle {i+1}: Ligne {validation['ligne']} "
                          f"(conf: {validation['confidence']:.2f}, "
                          f"dist: {validation['color_distance']:.3f})")
        
        # 3. Supprimer les doublons proches
        final_detections = self._remove_duplicate_detections(valid_detections)
        
        if debug:
            print(f"🎯 DÉTECTIONS FINALES: {len(final_detections)}")
        
        return final_detections
    
    def _detect_circles(self, image):
        """
        Détecte les cercles avec approche équilibrée (ni trop strict, ni trop permissif)
        """
        # Convertir en niveaux de gris
        gray = color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        # Prétraitement modéré
        gray_blurred = cv2.medianBlur(gray_uint8, 5)
        
        # Configuration équilibrée - plus permissive que l'ultra-strict
        configurations = [
            # Configuration principale - équilibrée
            {'dp': 1, 'minDist': 30, 'param1': 60, 'param2': 40, 'minRadius': 16, 'maxRadius': 55},
            # Configuration backup - un peu plus permissive
            {'dp': 1, 'minDist': 25, 'param1': 50, 'param2': 35, 'minRadius': 14, 'maxRadius': 60}
        ]
        
        all_circles = []
        
        for config in configurations:
            circles = cv2.HoughCircles(
                gray_blurred,
                cv2.HOUGH_GRADIENT,
                dp=config['dp'],
                minDist=config['minDist'],
                param1=config['param1'],
                param2=config['param2'],
                minRadius=config['minRadius'],
                maxRadius=config['maxRadius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for x, y, r in circles:
                    # Filtrage basique seulement
                    h, w = image.shape[:2]
                    margin = 20
                    if (margin < x < w-margin and margin < y < h-margin and 
                        14 <= r <= 60):
                        
                        # Validation légère - juste s'assurer que c'est coloré
                        if self._basic_color_validation(image, x, y, r):
                            all_circles.append((x, y, r))
        
        # Fusionner les cercles similaires
        unique_circles = self._merge_similar_circles(all_circles)
        
        # Limiter raisonnablement (pas trop strict)
        if len(unique_circles) > 25:
            print(f"⚠️ Beaucoup de candidats ({len(unique_circles)}), tri par qualité...")
            scored_circles = []
            for x, y, r in unique_circles:
                score = self._simple_quality_score(image, x, y, r)
                scored_circles.append((x, y, r, score))
            
            scored_circles.sort(key=lambda item: item[3], reverse=True)
            unique_circles = [(x, y, r) for x, y, r, _ in scored_circles[:20]]
        
        return unique_circles
    
    def _basic_color_validation(self, image, x, y, r):
        """
        Validation basique - juste vérifier qu'il y a de la couleur
        """
        # Extraire région
        margin = 5
        x1, x2 = max(0, x-r-margin), min(image.shape[1], x+r+margin)
        y1, y2 = max(0, y-r-margin), min(image.shape[0], y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        region = image[y1:y2, x1:x2]
        
        # Simple test: il faut au moins 15% de pixels avec un minimum de couleur
        hsv_region = color.rgb2hsv(region)
        saturation = hsv_region[:, :, 1]
        
        colorful_pixels = np.sum(saturation > 0.1)  # Seuil très bas
        total_pixels = saturation.size
        
        return (colorful_pixels / total_pixels) > 0.15
    
    def _simple_quality_score(self, image, x, y, r):
        """
        Score simple pour trier les cercles
        """
        margin = 3
        x1, x2 = max(0, x-r-margin), min(image.shape[1], x+r+margin)
        y1, y2 = max(0, y-r-margin), min(image.shape[0], y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return 0
            
        region = image[y1:y2, x1:x2]
        h, w = region.shape[:2]
        
        score = 0
        
        # 1. Saturation moyenne (plus coloré = mieux)
        hsv_region = color.rgb2hsv(region)
        avg_saturation = np.mean(hsv_region[:, :, 1])
        score += avg_saturation * 3
        
        # 2. Circularité basique
        circularity = 1.0 - abs(h - w) / max(h, w)
        score += circularity * 2
        
        # 3. Taille raisonnable (ni trop petit ni trop grand)
        size_score = 1.0 - abs(r - 30) / 30  # Optimal autour de 30px
        score += max(0, size_score)
        
        return score
    
    def _validate_metro_sign_candidate(self, image, x, y, r):
        """
        Validation spécifique pour candidat signe métro (bordure épaisse colorée)
        """
        # Extraire région
        margin = 8
        x1, x2 = max(0, x-r-margin), min(image.shape[1], x+r+margin)
        y1, y2 = max(0, y-r-margin), min(image.shape[0], y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        region = image[y1:y2, x1:x2]
        h, w = region.shape[:2]
        center_y, center_x = h//2, w//2
        
        # Test 1: Bordure épaisse colorée
        if not self._has_thick_colored_border(region, center_x, center_y, r):
            return False
        
        # Test 2: Ratio de circularité (éviter les ellipses)
        if abs(h - w) > min(h, w) * 0.3:  # Trop elliptique
            return False
        
        # Test 3: Densité de couleur suffisante
        hsv_region = color.rgb2hsv(region)
        saturation = hsv_region[:, :, 1]
        colorful_ratio = np.sum(saturation > 0.2) / saturation.size
        
        return colorful_ratio > 0.3  # Au moins 30% de pixels colorés
    
    def _score_metro_sign_quality(self, image, x, y, r):
        """
        Score spécifique pour la qualité d'un signe métro
        """
        margin = 5
        x1, x2 = max(0, x-r-margin), min(image.shape[1], x+r+margin)
        y1, y2 = max(0, y-r-margin), min(image.shape[0], y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return 0
            
        region = image[y1:y2, x1:x2]
        h, w = region.shape[:2]
        center_y, center_x = h//2, w//2
        
        score = 0
        
        # 1. Score bordure épaisse (40% du score)
        if self._has_thick_colored_border(region, center_x, center_y, r):
            score += 4.0
        
        # 2. Score circularité (20% du score)
        circularity = 1.0 - abs(h - w) / max(h, w)
        score += circularity * 2.0
        
        # 3. Score saturation (25% du score)
        hsv_region = color.rgb2hsv(region)
        avg_saturation = np.mean(hsv_region[:, :, 1])
        score += avg_saturation * 2.5
        
        # 4. Score contraste centre/bordure (15% du score)
        # Centre doit être plus clair (chiffre blanc)
        center_region = region[center_y-r//3:center_y+r//3, center_x-r//3:center_x+r//3]
        if center_region.size > 0:
            center_brightness = np.mean(color.rgb2gray(center_region))
            border_pixels = self._sample_ring(region, center_x, center_y, r*0.7, r*0.9)
            if border_pixels:
                border_brightness = np.mean(color.rgb2gray(np.array(border_pixels)))
                contrast = center_brightness - border_brightness
                score += max(0, contrast) * 1.5
        
        return score
    
    def _validate_circle_shape(self, image, x, y, r):
        """
        Valide qu'un cercle détecté a bien une forme et couleur cohérente + bordure épaisse
        """
        # Extraire la région du cercle
        margin = 8
        x1, x2 = max(0, x-r-margin), min(image.shape[1], x+r+margin)
        y1, y2 = max(0, y-r-margin), min(image.shape[0], y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        region = image[y1:y2, x1:x2]
        h, w = region.shape[:2]
        center_y, center_x = h//2, w//2
        
        # Test 1: Vérifier qu'il y a une bordure épaisse colorée
        if not self._has_thick_colored_border(region, center_x, center_y, r):
            return False
        
        # Test 2: Vérifier qu'il y a suffisamment de couleur (pas juste du blanc/gris)
        hsv_region = color.rgb2hsv(region)
        saturation = hsv_region[:, :, 1]
        
        # Au moins 25% de pixels avec saturation > 0.2 (plus strict pour les signes métro)
        colorful_pixels = np.sum(saturation > 0.2)
        total_pixels = saturation.size
        
        return (colorful_pixels / total_pixels) > 0.25
    
    def _has_thick_colored_border(self, region, center_x, center_y, r):
        """
        Vérifie qu'il y a une bordure épaisse colorée caractéristique des signes métro
        """
        h, w = region.shape[:2]
        
        # Analyser plusieurs anneaux concentriques
        # Anneau externe (bordure du signe)
        outer_ring = self._sample_ring(region, center_x, center_y, r * 0.85, r * 0.95)
        # Anneau moyen (milieu de la bordure)  
        middle_ring = self._sample_ring(region, center_x, center_y, r * 0.7, r * 0.8)
        # Anneau interne (vers le centre/chiffre)
        inner_ring = self._sample_ring(region, center_x, center_y, r * 0.4, r * 0.6)
        
        if len(outer_ring) < 8 or len(middle_ring) < 8:
            return False
        
        # Convertir en HSV pour analyser
        outer_hsv = color.rgb2hsv(np.array(outer_ring).reshape(1, -1, 3))[0]
        middle_hsv = color.rgb2hsv(np.array(middle_ring).reshape(1, -1, 3))[0]
        
        # Critères pour une bordure épaisse de signe métro:
        # 1. Saturation élevée sur la bordure (couleur vive)
        outer_sat = np.mean(outer_hsv[:, 1])
        middle_sat = np.mean(middle_hsv[:, 1])
        
        # 2. Homogénéité de couleur sur la bordure (même teinte)
        outer_hue_std = np.std(outer_hsv[:, 0])
        middle_hue_std = np.std(middle_hsv[:, 0])
        
        # 3. Contraste avec le centre (chiffre blanc)
        if len(inner_ring) > 4:
            inner_hsv = color.rgb2hsv(np.array(inner_ring).reshape(1, -1, 3))[0]
            inner_value = np.mean(inner_hsv[:, 2])  # Luminosité
            border_value = np.mean(middle_hsv[:, 2])
            value_contrast = abs(inner_value - border_value)
        else:
            value_contrast = 0.3  # Valeur par défaut
        
        # Critères de validation
        has_saturated_border = (outer_sat > 0.3 and middle_sat > 0.25)
        has_homogeneous_color = (outer_hue_std < 0.15 and middle_hue_std < 0.15)
        has_good_contrast = value_contrast > 0.2
        
        return has_saturated_border and has_homogeneous_color and has_good_contrast
    
    def _sample_ring(self, region, center_x, center_y, inner_radius, outer_radius):
        """
        Échantillonne les pixels dans un anneau entre inner_radius et outer_radius
        """
        h, w = region.shape[:2]
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        ring_mask = (distances >= inner_radius) & (distances <= outer_radius)
        
        if np.any(ring_mask):
            return region[ring_mask].tolist()
        else:
            return []
    
    def _score_circle_quality(self, image, x, y, r):
        """
        Score la qualité d'un cercle détecté
        """
        # Extraire région
        margin = 3
        x1, x2 = max(0, x-r-margin), min(image.shape[1], x+r+margin)
        y1, y2 = max(0, y-r-margin), min(image.shape[0], y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return 0
            
        region = image[y1:y2, x1:x2]
        
        # Critères de qualité
        # 1. Homogénéité de couleur sur le contour
        h, w = region.shape[:2]
        center_y, center_x = h//2, w//2
        
        # Échantillonner le contour du cercle
        angles = np.linspace(0, 2*np.pi, 16)
        contour_colors = []
        
        for angle in angles:
            px = int(center_x + r * 0.8 * np.cos(angle))
            py = int(center_y + r * 0.8 * np.sin(angle))
            
            if 0 <= px < w and 0 <= py < h:
                contour_colors.append(region[py, px])
        
        if len(contour_colors) < 8:
            return 0
            
        contour_colors = np.array(contour_colors)
        
        # 2. Variance de couleur sur le contour (plus faible = mieux)
        color_variance = np.mean(np.var(contour_colors, axis=0))
        
        # 3. Saturation moyenne (signes métro sont colorés)
        hsv_region = color.rgb2hsv(region)
        avg_saturation = np.mean(hsv_region[:, :, 1])
        
        # Score composite (plus haut = meilleur)
        quality_score = avg_saturation * 2 - color_variance * 5
        
        return max(0, quality_score)
    
    def _merge_similar_circles(self, circles):
        """
        Fusionne les cercles similaires détectés par différentes configurations
        """
        if not circles:
            return []
        
        merged = []
        used = set()
        
        for i, (x1, y1, r1) in enumerate(circles):
            if i in used:
                continue
                
            # Grouper les cercles similaires
            group = [(x1, y1, r1)]
            used.add(i)
            
            for j, (x2, y2, r2) in enumerate(circles[i+1:], i+1):
                if j in used:
                    continue
                
                # Distance entre centres
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                radius_diff = abs(r1-r2)
                
                # Si suffisamment proches, les fusionner
                if distance < 25 and radius_diff < 15:
                    group.append((x2, y2, r2))
                    used.add(j)
            
            # Calculer la médiane du groupe
            if len(group) > 1:
                xs, ys, rs = zip(*group)
                merged_circle = (int(np.median(xs)), int(np.median(ys)), int(np.median(rs)))
            else:
                merged_circle = group[0]
            
            merged.append(merged_circle)
        
        return merged
    
    def _validate_circle_by_color(self, image, x, y, r, colors_dict, tolerances_dict):
        """
        Validation couleur équilibrée - ni trop stricte ni trop permissive
        """
        # Extraire la région autour du cercle
        margin = 5
        x1, x2 = max(0, x-r-margin), min(image.shape[1], x+r+margin)
        y1, y2 = max(0, y-r-margin), min(image.shape[0], y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return {'is_valid': False}
        
        region = image[y1:y2, x1:x2]
        
        # Extraire couleur avec méthode équilibrée
        dominant_color = self._extract_balanced_color(region)
        
        # Comparer avec toutes les couleurs de lignes
        best_match = {'ligne': -1, 'confidence': 0.0, 'distance': float('inf')}
        second_best_distance = float('inf')
        
        for ligne, target_color in colors_dict.items():
            # Distance euclidienne dans l'espace RGB
            color_distance = np.linalg.norm(dominant_color - target_color)
            
            # Tolérance ajustée - plus permissive que le mode ultra-strict
            tolerance = tolerances_dict.get(ligne, 0.15) + 0.05  # Ajouter 0.05 de marge
            
            # Score de confiance
            if color_distance < tolerance:
                confidence = max(0, 1.0 - (color_distance / tolerance))
                
                if color_distance < best_match['distance']:
                    second_best_distance = best_match['distance']
                    best_match = {
                        'ligne': ligne,
                        'confidence': confidence,
                        'distance': color_distance
                    }
                elif color_distance < second_best_distance:
                    second_best_distance = color_distance
        
        # Validation plus permissive:
        # 1. Confiance moins stricte
        # 2. Discrimination moins exigeante
        discrimination_ratio = second_best_distance / best_match['distance'] if best_match['distance'] > 0 else float('inf')
        
        is_valid = (best_match['confidence'] > 0.4 and  # Confiance modérée (était 0.6)
                   discrimination_ratio > 1.1)  # Discrimination plus souple (était 1.3)
        
        result = {
            'is_valid': is_valid,
            'ligne': best_match['ligne'] if is_valid else -1,
            'confidence': best_match['confidence'],
            'color_distance': best_match['distance'],
            'bbox': (x1, x2, y1, y2) if is_valid else None
        }
        
        return result
    
    def _extract_balanced_color(self, region):
        """
        Extraction couleur équilibrée - compromise entre précision et robustesse
        """
        h, w = region.shape[:2]
        
        # Méthode 1: Focus sur les pixels moyennement saturés (pas blanc, pas noir)
        hsv_region = color.rgb2hsv(region)
        saturation = hsv_region[:, :, 1]
        value = hsv_region[:, :, 2]
        
        # Masque pour pixels colorés mais pas extrêmes
        good_mask = (saturation > 0.15) & (value > 0.1) & (value < 0.9)
        
        if np.sum(good_mask) > region.size * 0.1:  # Au moins 10% de bons pixels
            good_pixels = region[good_mask]
            
            # Si assez de pixels, faire un clustering simple
            if len(good_pixels) > 15:
                try:
                    kmeans = KMeans(n_clusters=min(2, len(good_pixels)//10), 
                                  random_state=42, n_init=10)
                    kmeans.fit(good_pixels)
                    
                    # Prendre le cluster le plus saturé
                    cluster_colors = kmeans.cluster_centers_
                    cluster_hsv = color.rgb2hsv(cluster_colors.reshape(1, -1, 3))[0]
                    
                    best_cluster_idx = np.argmax(cluster_hsv[:, 1])
                    return cluster_colors[best_cluster_idx]
                except:
                    pass
            
            # Fallback: moyenne des bons pixels
            return np.mean(good_pixels, axis=0)
        
        # Méthode 2: Bordure externe si pas assez de pixels colorés
        center_y, center_x = h//2, w//2
        radius = min(h, w) // 2
        
        # Échantillonner le contour externe
        border_pixels = self._sample_ring(region, center_x, center_y, radius*0.6, radius*0.9)
        
        if len(border_pixels) > 5:
            return np.mean(border_pixels, axis=0)
        
        # Fallback final: moyenne générale en évitant les extrêmes
        flat_region = region.reshape(-1, 3)
        # Éviter pixels trop blancs ou trop noirs
        flat_hsv = color.rgb2hsv(flat_region.reshape(1, -1, 3))[0]
        moderate_mask = (flat_hsv[:, 2] > 0.1) & (flat_hsv[:, 2] < 0.9)
        
        if np.any(moderate_mask):
            return np.mean(flat_region[moderate_mask], axis=0)
        else:
            return np.mean(flat_region, axis=0)
    
    def _extract_dominant_color_v2(self, region):
        """
        Version améliorée d'extraction de couleur dominante - Focus sur la bordure épaisse
        """
        h, w = region.shape[:2]
        center_y, center_x = h//2, w//2
        radius = min(h, w) // 2
        
        # MÉTHODE SPÉCIFIQUE MÉTRO: Analyser la bordure épaisse
        # Les signes métro ont une bordure colorée épaisse avec chiffre blanc au centre
        
        # 1. Extraire la bordure épaisse (zone colorée principale)
        border_colors = []
        
        # Échantillonner plusieurs anneaux de la bordure
        for ring_ratio in [0.7, 0.75, 0.8, 0.85]:
            ring_radius = radius * ring_ratio
            ring_pixels = self._sample_ring(region, center_x, center_y, 
                                          ring_radius - 2, ring_radius + 2)
            if ring_pixels:
                border_colors.extend(ring_pixels)
        
        if len(border_colors) > 10:
            border_colors = np.array(border_colors)
            
            # Filtrer les pixels trop clairs (blanc du chiffre) ou trop sombres
            hsv_border = color.rgb2hsv(border_colors.reshape(1, -1, 3))[0]
            
            # Garder seulement les pixels avec bonne saturation et luminosité modérée
            good_mask = (hsv_border[:, 1] > 0.25) & (hsv_border[:, 2] > 0.2) & (hsv_border[:, 2] < 0.8)
            
            if np.sum(good_mask) > 5:
                good_border_colors = border_colors[good_mask]
                
                # Clustering pour trouver LA couleur dominante de la bordure
                if len(good_border_colors) > 8:
                    try:
                        # Utiliser 2-3 clusters pour capturer la couleur principale
                        n_clusters = min(3, len(good_border_colors)//8)
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        kmeans.fit(good_border_colors)
                        
                        # Prendre le cluster le plus saturé (= couleur du signe)
                        cluster_colors = kmeans.cluster_centers_
                        cluster_hsv = color.rgb2hsv(cluster_colors.reshape(1, -1, 3))[0]
                        
                        # Choisir le cluster avec la meilleure saturation
                        best_cluster_idx = np.argmax(cluster_hsv[:, 1])
                        return cluster_colors[best_cluster_idx]
                    except:
                        pass
                
                # Fallback: moyenne des bons pixels de bordure
                return np.mean(good_border_colors, axis=0)
        
        # Méthode 2: Si pas assez de bordure détectée, méthode classique mais plus stricte
        hsv_region = color.rgb2hsv(region)
        saturation = hsv_region[:, :, 1]
        value = hsv_region[:, :, 2]
        
        # Masque pour pixels colorés modérément (ni blanc ni noir)
        colored_mask = (saturation > 0.3) & (value > 0.15) & (value < 0.85)
        
        if np.sum(colored_mask) > region.size * 0.15:  # Au moins 15% de pixels colorés
            colored_pixels = region[colored_mask]
            return np.mean(colored_pixels, axis=0)
        
        # Fallback final: moyenne des pixels non-blancs
        non_white_mask = value < 0.9
        if np.any(non_white_mask):
            return np.mean(region[non_white_mask], axis=0)
        
        return np.mean(region.reshape(-1, 3), axis=0)
    
    def _remove_duplicate_detections(self, detections):
        """
        Supprime les détections dupliquées (cercles proches)
        """
        if not detections:
            return detections
        
        # Trier par confiance décroissante
        detections_sorted = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        
        final_detections = []
        
        for detection in detections_sorted:
            x1, y1 = detection['center']
            
            # Vérifier si une détection similaire existe déjà
            is_duplicate = False
            for existing in final_detections:
                x2, y2 = existing['center']
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                # Si trop proche d'une détection existante
                if distance < 40:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        return final_detections


def processOneMetroImage(nom, im, n, resizeFactor, save_images=False, metro_system=None):
    """
    Fonction principale de traitement d'une image (compatible avec l'interface existante)
    """
    # Redimensionner l'image si nécessaire
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
    print(f"Traitement image: {nom}")
    print(f"{'='*60}")
    
    # Utiliser le système fourni ou créer un nouveau
    if metro_system is None:
        metro_system = ImprovedMetroSystem()
        print("⚠️ Utilisation du système non-entraîné")
    
    # Détecter les signes de métro
    detections = metro_system.detect_metro_signs(im_resized, debug=True)
    
    # Convertir au format attendu: [n, x1, x2, y1, y2, ligne]
    bd = []
    for detection in detections:
        if detection['bbox']:
            x1, x2, y1, y2 = detection['bbox']
            # Note: garder l'ordre des coordonnées pour compatibilité
            bd.append([n, y1, y2, x1, x2, detection['ligne']])
    
    bd = np.array(bd) if bd else np.empty((0, 6))
    
    # Affichage des résultats
    plt.figure(figsize=(16, 10))
    plt.imshow(im_resized)
    
    # Couleurs pour l'affichage
    colors = ['lime', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
    for i, detection in enumerate(detections):
        if detection['bbox']:
            x1, x2, y1, y2 = detection['bbox']
            
            # Rectangle de détection
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=3, 
                           edgecolor=colors[i % len(colors)], 
                           facecolor='none')
            plt.gca().add_patch(rect)
            
            # Cercle détecté
            circle = Circle(detection['center'], detection['radius'],
                          linewidth=2, 
                          edgecolor=colors[i % len(colors)], 
                          facecolor='none',
                          linestyle='--')
            plt.gca().add_patch(circle)
            
            # Label avec numéro de ligne
            plt.text(x1, y1-10, f"L{detection['ligne']}", 
                    color='white', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=colors[i % len(colors)], alpha=0.8))
            
            # Confiance
            plt.text(x2, y2+15, f"Conf: {detection['confidence']:.2f}", 
                    color='white', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", 
                            facecolor='black', alpha=0.7))
    
    # Titre avec résumé
    mode = "ENTRAÎNÉ" if metro_system.is_trained else "THÉORIQUE"
    titre = f"{nom} - {len(detections)} détection(s) ({mode})"
    if detections:
        lignes_detectees = [str(d['ligne']) for d in detections]
        titre += f"\nLignes: {', '.join(lignes_detectees)}"
    
    plt.title(titre, fontsize=12, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"🎯 Résultat: {len(detections)} signe(s) détecté(s)")
    for det in detections:
        print(f"  ✅ Ligne {det['ligne']} (confiance: {det['confidence']:.3f})")
    
    return im_resized, bd


# Classe compatible avec l'ancien code
FinalMetroSystem = ImprovedMetroSystem

if __name__ == "__main__":
    print("Système de détection de métro amélioré prêt.")
    print("Usage:")
    print("1. system = ImprovedMetroSystem()")
    print("2. system.train_system('../BD_METRO', 'Apprentissage.mat')")
    print("3. detections = system.detect_metro_signs(image)")