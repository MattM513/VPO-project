# -*- coding: utf-8 -*-
"""
Version améliorée du détecteur de signes de métro
Corrections principales :
1. Couleurs de référence mises à jour
2. Amélioration de l'extraction de couleur
3. Filtrage plus strict des cercles
4. Meilleure analyse du contraste
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage as ski
from skimage import segmentation, measure, morphology, filters, feature, color
from skimage.transform import resize
from scipy import ndimage
import cv2


class ImprovedMetroDetector:
    """
    Détecteur amélioré pour les signes de métro parisien
    """
    
    def __init__(self):
        # Couleurs officielles mises à jour (RGB normalisé)
        # Ajustées selon les vraies couleurs des panneaux
        self.ligne_colors = {
            1: [1.0, 0.8, 0.0],         # Jaune
            2: [0.0, 0.4, 0.8],         # Bleu
            3: [0.6, 0.6, 0.2],         # Vert olive
            4: [0.8, 0.2, 0.5],         # Rose/Magenta - CORRIGÉ
            5: [0.9, 0.5, 0.2],         # Orange
            6: [0.5, 0.8, 0.5],         # Vert clair
            7: [0.9, 0.6, 0.7],         # Rose clair
            8: [0.8, 0.7, 0.8],         # Mauve clair
            9: [0.8, 0.8, 0.0],         # Jaune-vert
            10: [0.9, 0.7, 0.2],        # Jaune orangé
            11: [0.6, 0.4, 0.2],        # Marron
            12: [0.0, 0.6, 0.3],        # Vert
            13: [0.6, 0.8, 0.9],        # Bleu clair
            14: [0.4, 0.2, 0.6]         # Violet
        }
        
        self.confidence_threshold = 0.6  # Seuil plus strict
    
    def detect_metro_signs_improved(self, image):
        """
        Détection améliorée avec filtrage plus strict
        """
        candidates = []
        
        # Détection de cercles avec paramètres plus stricts
        circles = self._detect_circular_shapes_strict(image)
        print(f"Cercles détectés après filtrage strict: {len(circles)}")
        
        # Validation pour chaque cercle
        for circle in circles:
            x, y, r = circle
            
            # Extraction de la région avec marge
            margin = 5
            x1 = max(0, x - r - margin)
            x2 = min(image.shape[1], x + r + margin)
            y1 = max(0, y - r - margin)
            y2 = min(image.shape[0], y + r + margin)
            
            if x2 > x1 and y2 > y1:
                region = image[y1:y2, x1:x2]
                validation_result = self._validate_metro_sign_improved(region, r)
                
                if validation_result['is_valid']:
                    candidates.append({
                        'bbox': (x1, x2, y1, y2),
                        'area': np.pi * r * r,
                        'circularity': 1.0,
                        'centroid': (y, x),
                        'radius': r,
                        'predicted_line': validation_result['predicted_line'],
                        'confidence': validation_result['confidence'],
                        'method': 'improved',
                        'color_match': validation_result['color_match'],
                        'contrast_score': validation_result['contrast_score']
                    })
        
        return candidates
    
    def _detect_circular_shapes_strict(self, image):
        """
        Détection de cercles avec paramètres plus stricts pour réduire les faux positifs
        """
        gray = ski.color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        circles = []
        
        try:
            # Paramètres plus stricts
            detected_circles = cv2.HoughCircles(
                gray_uint8,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,        # Distance minimum augmentée
                param1=80,         # Seuil plus strict pour la détection des contours
                param2=40,         # Seuil plus strict pour le centre
                minRadius=25,      # Rayon minimum augmenté
                maxRadius=50       # Rayon maximum réduit
            )
            
            if detected_circles is not None:
                detected_circles = np.round(detected_circles[0, :]).astype("int")
                print(f"Cercles bruts détectés: {len(detected_circles)}")
                
                # Filtrage géographique vers le centre de l'image
                img_center_x = image.shape[1] / 2
                img_center_y = image.shape[0] / 2
                max_dist = min(image.shape[0], image.shape[1]) / 3  # Zone plus restreinte
                
                filtered_circles = []
                for x, y, r in detected_circles:
                    dist = np.sqrt((x - img_center_x)**2 + (y - img_center_y)**2)
                    if dist < max_dist:
                        filtered_circles.append((x, y, r))
                
                print(f"Cercles après filtrage géographique: {len(filtered_circles)}")
                
                # Limiter à 10 cercles maximum et trier par taille
                filtered_circles.sort(key=lambda c: c[2], reverse=True)  # Trier par rayon
                circles = filtered_circles[:10]
                print(f"Cercles gardés pour validation: {len(circles)}")
                
        except Exception as e:
            print(f"Erreur détection cercles: {e}")
        
        return circles
    
    def _validate_metro_sign_improved(self, region, radius):
        """
        Validation améliorée avec meilleure extraction de couleur et contraste
        """
        if region.size == 0:
            return {'is_valid': False, 'predicted_line': 1, 'confidence': 0.0}
        
        print(f"    DEBUG: Validation région {region.shape}")
        
        # Analyses améliorées
        color_analysis = self._analyze_color_improved(region)
        print(f"    Couleur: match={color_analysis['best_match']}, score={color_analysis['score']:.3f}")
        
        contrast_analysis = self._analyze_contrast_improved(region)
        print(f"    Contraste: has_contrast={contrast_analysis['has_contrast']}, score={contrast_analysis['score']:.3f}")
        
        structure_analysis = self._analyze_structure_improved(region)
        print(f"    Structure: score={structure_analysis['score']:.3f}")
        
        # Score global avec pondération ajustée
        total_score = (
            color_analysis['score'] * 0.5 +      # 50% couleur (plus important)
            contrast_analysis['score'] * 0.3 +   # 30% contraste
            structure_analysis['score'] * 0.2    # 20% structure
        )
        
        print(f"    SCORE TOTAL: {total_score:.3f}")
        
        # Validation plus stricte
        is_valid = (
            total_score > self.confidence_threshold and
            color_analysis['best_match'] != -1 and
            contrast_analysis['has_contrast'] and
            structure_analysis['score'] > 0.7
        )
        
        print(f"    VALIDÉ: {is_valid}")
        
        return {
            'is_valid': is_valid,
            'predicted_line': color_analysis['best_match'] if color_analysis['best_match'] != -1 else 1,
            'confidence': total_score,
            'color_match': color_analysis['score'],
            'contrast_score': contrast_analysis['score']
        }
    
    def _analyze_color_improved(self, region):
        """
        Analyse de couleur améliorée - extraction du cercle externe
        """
        h, w = region.shape[:2]
        center_y, center_x = h//2, w//2
        
        # Créer un masque circulaire pour extraire la couleur du bord
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        radius = min(h, w) // 2
        # Zone annulaire externe (couleur de la ligne)
        ring_mask = (distances >= radius * 0.7) & (distances <= radius * 0.95)
        
        if np.any(ring_mask):
            ring_pixels = region[ring_mask]
            # Utiliser la médiane pour être plus robuste aux outliers
            dominant_color = np.median(ring_pixels, axis=0)
        else:
            # Fallback : couleur moyenne de la région
            dominant_color = np.mean(region.reshape(-1, 3), axis=0)
        
        # Comparaison avec les couleurs de référence
        best_match = -1
        min_distance = float('inf')
        
        for ligne, ref_color in self.ligne_colors.items():
            distance = np.linalg.norm(dominant_color - np.array(ref_color))
            if distance < min_distance:
                min_distance = distance
                best_match = ligne
        
        # Score basé sur la distance (plus strict)
        score = max(0, 1.0 - min_distance / 0.3)  # Seuil plus strict
        
        # Invalider si trop loin de toute couleur de référence
        if min_distance > 0.3:
            best_match = -1
            score = 0.0
        
        return {
            'best_match': best_match,
            'score': score,
            'distance': min_distance,
            'dominant_color': dominant_color
        }
    
    def _analyze_contrast_improved(self, region):
        """
        Analyse du contraste améliorée
        """
        if region.size == 0:
            return {'has_contrast': False, 'score': 0.0}
        
        gray = ski.color.rgb2gray(region)
        h, w = gray.shape
        center_y, center_x = h//2, w//2
        
        # Zone centrale plus petite (chiffre)
        radius_center = min(h, w) // 6  # Plus petit
        
        if radius_center < 2:
            return {'has_contrast': False, 'score': 0.0}
        
        # Masque pour la zone centrale
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        center_mask = distances <= radius_center
        peripheral_mask = (distances >= radius_center * 2) & (distances <= min(h, w) // 2)
        
        if np.any(center_mask) and np.any(peripheral_mask):
            center_brightness = np.mean(gray[center_mask])
            peripheral_brightness = np.mean(gray[peripheral_mask])
            
            contrast = abs(center_brightness - peripheral_brightness)
            # Le centre (chiffre) doit être plus clair que la périphérie (fond coloré)
            good_pattern = center_brightness > peripheral_brightness + 0.1
            
            has_contrast = contrast > 0.3 and good_pattern
            score = contrast if has_contrast else 0.0
        else:
            has_contrast = False
            score = 0.0
            contrast = 0.0
            center_brightness = 0.0
            peripheral_brightness = 0.0
        
        return {
            'has_contrast': has_contrast,
            'score': min(score, 1.0),
            'contrast_value': contrast,
            'center_brightness': center_brightness,
            'peripheral_brightness': peripheral_brightness
        }
    
    def _analyze_structure_improved(self, region):
        """
        Analyse de structure améliorée
        """
        if region.size == 0:
            return {'score': 0.0}
        
        h, w = region.shape[:2]
        
        # Vérification du ratio d'aspect
        aspect_ratio = min(w/h, h/w) if h > 0 and w > 0 else 0
        
        # Vérification de la taille
        size = min(h, w)
        size_ok = 40 <= size <= 100  # Taille attendue pour un signe de métro
        
        # Score combiné
        score = aspect_ratio if (aspect_ratio > 0.8 and size_ok) else 0.0
        
        return {
            'score': score,
            'aspect_ratio': aspect_ratio,
            'size': size,
            'size_ok': size_ok
        }


def processOneMetroImage(nom, im, n, resizeFactor, save_images=False):
    """
    Fonction de traitement améliorée
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
    
    # Initialisation du détecteur amélioré
    detector = ImprovedMetroDetector()
    
    # Prétraitement adaptatif
    image_enhanced = ski.exposure.equalize_adapthist(im_resized, clip_limit=0.03)
    
    # Détection améliorée
    candidates = detector.detect_metro_signs_improved(image_enhanced)
    print(f"Image {nom}: {len(candidates)} candidats trouvés")
    
    # Debug des candidats
    for i, candidate in enumerate(candidates):
        bbox = candidate['bbox']
        ligne = candidate.get('predicted_line', 'unknown')
        confidence = candidate.get('confidence', 0)
        
        x1, x2, y1, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        print(f"  Candidat {i+1}: Ligne {ligne}, Centre ({center_x:.0f},{center_y:.0f})")
        print(f"    Confiance: {confidence:.3f}")
    
    # Filtrage final par confiance et limitation
    confident_candidates = [c for c in candidates if c.get('confidence', 0) >= detector.confidence_threshold]
    
    if len(confident_candidates) > 3:
        confident_candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        confident_candidates = confident_candidates[:3]
    
    # Construction du résultat
    bd = []
    for candidate in confident_candidates:
        x1, x2, y1, y2 = candidate['bbox']
        ligne = candidate['predicted_line']
        bd.append([n, x1, x2, y1, y2, ligne])
    
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
    
    return im_resized, bd


def draw_rectangle(x1, x2, y1, y2, color):
    """Dessine un rectangle sur le graphique actuel"""
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                    linewidth=2, edgecolor=color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)