# -*- coding: utf-8 -*-
"""
Système de reconnaissance automatique des lignes de métro parisien
Détection structurelle basée sur forme, couleur et contraste
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage as ski
from skimage import segmentation, measure, morphology, filters, feature, color
from skimage.transform import resize
from scipy import ndimage
import cv2


class StructuralMetroDetector:
    """
    Détecteur basé sur la structure complète des signes de métro
    """
    
    def __init__(self):
        # Couleurs officielles des lignes de métro (RGB normalisé)
        self.ligne_colors = {
            1: [1.0, 0.808, 0.0],       # #FFCE00 - Jaune
            2: [0.0, 0.392, 0.690],     # #0064B0 - Bleu
            3: [0.624, 0.596, 0.145],   # #9F9825 - Vert olive
            4: [0.753, 0.255, 0.569],   # #C04191 - Rose/Magenta
            5: [0.949, 0.557, 0.259],   # #F28E42 - Orange
            6: [0.514, 0.769, 0.569],   # #83C491 - Vert clair
            7: [0.953, 0.643, 0.729],   # #F3A4BA - Rose clair
            8: [0.808, 0.678, 0.824],   # #CEADD2 - Mauve clair
            9: [0.835, 0.788, 0.0],     # #D5C900 - Jaune-vert
            10: [0.890, 0.702, 0.165],  # #E3B32A - Jaune orangé
            11: [0.553, 0.369, 0.165],  # #8D5E2A - Marron
            12: [0.0, 0.506, 0.310],    # #00814F - Vert
            13: [0.596, 0.831, 0.886],  # #98D4E2 - Bleu clair
            14: [0.400, 0.145, 0.514]   # #662483 - Violet
        }
        
        self.confidence_threshold = 0.3
    
    def detect_metro_signs_structural(self, image):
        """
        Détection basée sur la structure : Forme + Couleur + Contenu
        """
        candidates = []
        
        # Détection de cercles par Hough Transform
        circles = self._detect_circular_shapes(image)
        print(f"Cercles détectés: {len(circles)}")
        
        # Validation structurelle pour chaque cercle
        for circle in circles:
            x, y, r = circle
            
            x1 = max(0, x - r)
            x2 = min(image.shape[1], x + r)
            y1 = max(0, y - r)
            y2 = min(image.shape[0], y + r)
            
            if x2 > x1 and y2 > y1:
                region = image[y1:y2, x1:x2]
                validation_result = self._validate_metro_sign_structure(region, r)
                
                if validation_result['is_valid']:
                    candidates.append({
                        'bbox': (x1, x2, y1, y2),
                        'area': np.pi * r * r,
                        'circularity': 1.0,
                        'centroid': (y, x),
                        'radius': r,
                        'predicted_line': validation_result['predicted_line'],
                        'confidence': validation_result['confidence'],
                        'method': 'structural',
                        'color_match': validation_result['color_match'],
                        'contrast_score': validation_result['contrast_score']
                    })
        
        return candidates
    
    def _detect_circular_shapes(self, image):
        """
        Détection de cercles optimisée pour les signes de métro
        """
        gray = ski.color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        circles = []
        
        try:
            detected_circles = cv2.HoughCircles(
                gray_uint8,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=25,
                minRadius=20,
                maxRadius=60
            )
            
            if detected_circles is not None:
                detected_circles = np.round(detected_circles[0, :]).astype("int")
                circles = [(x, y, r) for x, y, r in detected_circles]
                
        except Exception as e:
            print(f"Erreur détection cercles: {e}")
        
        return circles
    
    def _validate_metro_sign_structure(self, region, radius):
        """
        Validation complète de la structure d'un signe de métro
        """
        if region.size == 0:
            return {'is_valid': False, 'predicted_line': 1, 'confidence': 0.0}
        
        # Analyses multi-critères
        color_analysis = self._analyze_dominant_color(region)
        contrast_analysis = self._analyze_contrast_pattern(region)
        structure_analysis = self._analyze_circular_structure(region)
        size_analysis = self._analyze_size_coherence(region, radius)
        
        # Score global pondéré
        total_score = (
            color_analysis['score'] * 0.4 +      # 40% couleur
            contrast_analysis['score'] * 0.3 +   # 30% contraste
            structure_analysis['score'] * 0.2 +  # 20% structure
            size_analysis['score'] * 0.1         # 10% taille
        )
        
        # Validation finale
        is_valid = (
            total_score > 0.5 and
            color_analysis['best_match'] != -1 and
            contrast_analysis['has_contrast'] and
            structure_analysis['is_circular']
        )
        
        return {
            'is_valid': is_valid,
            'predicted_line': color_analysis['best_match'] if color_analysis['best_match'] != -1 else 1,
            'confidence': total_score,
            'color_match': color_analysis['score'],
            'contrast_score': contrast_analysis['score'],
            'structure_score': structure_analysis['score'],
            'size_score': size_analysis['score']
        }
    
    def _analyze_dominant_color(self, region):
        """
        Analyse de la couleur dominante avec comparaison précise
        """
        h, w = region.shape[:2]
        center = (h//2, w//2)
        
        # Masque annulaire pour extraire la couleur du bord du cercle
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center[1])**2 + (y_coords - center[0])**2)
        
        radius = min(h, w) // 2
        outer_ring = (distances >= radius * 0.6) & (distances <= radius * 0.9)
        
        if np.any(outer_ring):
            ring_color = np.mean(region[outer_ring], axis=0)
        else:
            ring_color = np.mean(region.reshape(-1, 3), axis=0)
        
        # Comparaison avec couleurs de lignes
        best_match = -1
        min_distance = float('inf')
        
        for ligne, ref_color in self.ligne_colors.items():
            distance = np.linalg.norm(ring_color - np.array(ref_color))
            if distance < min_distance:
                min_distance = distance
                best_match = ligne
        
        score = max(0, 1.0 - min_distance / 0.5)
        
        if min_distance > 0.4:
            best_match = -1
            score = 0.0
        
        return {
            'best_match': best_match,
            'score': score,
            'distance': min_distance,
            'dominant_color': ring_color
        }
    
    def _analyze_contrast_pattern(self, region):
        """
        Analyse du contraste (chiffre blanc sur fond coloré)
        """
        if region.size == 0:
            return {'has_contrast': False, 'score': 0.0}
        
        gray = ski.color.rgb2gray(region)
        h, w = gray.shape
        center_h, center_w = h//2, w//2
        radius_center = min(h, w) // 4
        
        if radius_center < 3:
            return {'has_contrast': False, 'score': 0.0}
        
        # Zone centrale (chiffre)
        center_region = gray[
            max(0, center_h-radius_center):center_h+radius_center,
            max(0, center_w-radius_center):center_w+radius_center
        ]
        
        # Zone périphérique (fond coloré)
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_w)**2 + (y_coords - center_h)**2)
        radius = min(h, w) // 2
        
        peripheral_mask = (distances >= radius * 0.5) & (distances <= radius * 0.9)
        
        if np.any(peripheral_mask):
            peripheral_region = gray[peripheral_mask]
        else:
            peripheral_region = gray.flatten()
        
        center_brightness = np.mean(center_region)
        peripheral_brightness = np.mean(peripheral_region)
        
        contrast = abs(center_brightness - peripheral_brightness)
        has_contrast = contrast > 0.2
        good_pattern = center_brightness > peripheral_brightness  # Centre plus clair
        
        score = contrast if has_contrast and good_pattern else 0.0
        
        return {
            'has_contrast': has_contrast and good_pattern,
            'score': min(score, 1.0),
            'contrast_value': contrast,
            'center_brightness': center_brightness,
            'peripheral_brightness': peripheral_brightness
        }
    
    def _analyze_circular_structure(self, region):
        """
        Analyse de la structure circulaire
        """
        if region.size == 0:
            return {'is_circular': False, 'score': 0.0}
        
        h, w = region.shape[:2]
        aspect_ratio = min(w/h, h/w) if h > 0 and w > 0 else 0
        size_ok = 30 <= min(h, w) <= 120
        
        is_circular = aspect_ratio > 0.8 and size_ok
        
        return {
            'is_circular': is_circular,
            'score': aspect_ratio if size_ok else 0.0,
            'aspect_ratio': aspect_ratio,
            'size_ok': size_ok
        }
    
    def _analyze_size_coherence(self, region, radius):
        """
        Analyse de la cohérence de taille
        """
        h, w = region.shape[:2]
        expected_size = radius * 2
        actual_size = min(h, w)
        
        size_diff = abs(actual_size - expected_size) / expected_size
        score = max(0, 1.0 - size_diff)
        size_ok = 30 <= actual_size <= 120
        
        return {
            'score': score if size_ok else 0.0,
            'size_coherent': size_diff < 0.3,
            'expected_size': expected_size,
            'actual_size': actual_size
        }


def processOneMetroImage(nom, im, n, resizeFactor, save_images=False):
    """
    Fonction principale de traitement avec détection structurelle
    """
    
    # Redimensionnement
    if resizeFactor != 1:
        im_resized = ski.transform.resize(
            im, 
            (int(im.shape[0] * resizeFactor), int(im.shape[1] * resizeFactor)),
            anti_aliasing=True, 
            preserve_range=True
        ).astype(im.dtype)
    else:
        im_resized = im
    
    print(f"\nTraitement image {nom} (taille: {im_resized.shape})")
    
    # Initialisation du détecteur structurel
    detector = StructuralMetroDetector()
    
    # Prétraitement
    image_enhanced = ski.exposure.equalize_adapthist(im_resized, clip_limit=0.02)
    image_processed = ski.filters.gaussian(image_enhanced, sigma=0.8)
    
    # Détection structurelle
    candidates = detector.detect_metro_signs_structural(image_processed)
    print(f"Image {nom}: {len(candidates)} candidats structurels trouvés")
    
    # Debug des candidats
    for i, candidate in enumerate(candidates):
        bbox = candidate['bbox']
        ligne = candidate.get('predicted_line', 'unknown')
        confidence = candidate.get('confidence', 0)
        color_match = candidate.get('color_match', 0)
        contrast = candidate.get('contrast_score', 0)
        
        x1, x2, y1, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        print(f"  Candidat {i+1}: Ligne {ligne}, Centre ({center_x:.0f},{center_y:.0f})")
        print(f"    Confiance globale: {confidence:.3f}")
        print(f"    Match couleur: {color_match:.3f}")
        print(f"    Score contraste: {contrast:.3f}")
    
    # Filtrage par confiance
    confident_candidates = []
    for candidate in candidates:
        confidence = candidate.get('confidence', 0)
        if confidence >= detector.confidence_threshold:
            confident_candidates.append(candidate)
    
    print(f"Candidats avec confiance >= {detector.confidence_threshold}: {len(confident_candidates)}")
    
    # Limitation à 3 détections maximum
    if len(confident_candidates) > 3:
        confident_candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        confident_candidates = confident_candidates[:3]
        print(f"Limité à 3 meilleures détections")
    
    # Construction du résultat final
    bd = []
    
    for i, candidate in enumerate(confident_candidates):
        x1, x2, y1, y2 = candidate['bbox']
        ligne = candidate['predicted_line']
        
        print(f"Détection finale {i+1}: Ligne {ligne}, Coords ({x1},{y1})-({x2},{y2})")
        bd.append([n, x1, x2, y1, y2, ligne])
    
    # Conversion en numpy array
    if bd:
        bd = np.array(bd)
        print(f"Résultat final: {len(bd)} détections")
    else:
        bd = np.empty((0, 6))
        print("Aucune détection finale")
    
    # Affichage des résultats
    plt.figure(figsize=(12, 8))
    plt.imshow(im_resized)

    if bd.size > 0:
        for k in range(bd.shape[0]):
            x1, y1, x2, y2 = int(bd[k,1]), int(bd[k,3]), int(bd[k,2]), int(bd[k,4])
            ligne = int(bd[k,5])
            
            draw_rectangle(x1, x2, y1, y2, 'g')
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            plt.text(center_x, center_y, str(ligne), 
                    color='red', fontsize=14, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        lignes_detectees = bd[:,5].astype(int)
        plt.title(f'{nom} - Lignes détectées: {lignes_detectees} ({len(lignes_detectees)} signes)', 
                 fontsize=16, fontweight='bold')
    else:
        plt.title(f'{nom} - Aucune ligne détectée', fontsize=16, fontweight='bold')

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Sauvegarde conditionnelle
    if save_images:
        import os
        output_dir = 'results_images'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{nom}_detected.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Image sauvegardée : {output_path}")
    
    return im_resized, bd


def draw_rectangle(x1, x2, y1, y2, color):
    """
    Dessine un rectangle sur le graphique actuel
    """
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                    linewidth=2, edgecolor=color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)