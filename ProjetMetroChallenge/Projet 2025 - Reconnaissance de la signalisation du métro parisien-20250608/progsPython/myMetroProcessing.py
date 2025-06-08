# -*- coding: utf-8 -*-
"""
Système de reconnaissance automatique des lignes de métro parisien - Version finale
Intégration complète: détection, reconnaissance et validation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage as ski
from skimage import segmentation, measure, morphology, filters, feature, color
from skimage.transform import resize
from scipy import ndimage
import cv2
from sklearn.cluster import DBSCAN, KMeans


class MetroLineDetector:
    """
    Système complet de détection et reconnaissance des lignes de métro
    """
    
    def __init__(self):
        # Paramètres de détection - optimisés pour les signes de métro
        self.min_area = 300
        self.max_area = 8000
        self.min_circularity = 0.3
        self.color_tolerance = 0.2
        
        # Couleurs de référence des lignes (RGB normalisé, calibrées)
        self.ligne_colors = {
            1: [0.9, 0.8, 0.1],     # Jaune
            2: [0.1, 0.3, 0.8],     # Bleu
            3: [0.4, 0.5, 0.2],     # Vert olive
            4: [0.5, 0.1, 0.7],     # Violet
            5: [0.9, 0.4, 0.1],     # Orange
            6: [0.3, 0.7, 0.5],     # Vert clair
            7: [0.9, 0.3, 0.5],     # Rose
            8: [0.5, 0.7, 0.9],     # Bleu clair
            9: [0.1, 0.4, 0.1],     # Vert foncé
            10: [0.5, 0.3, 0.1],    # Marron
            11: [0.6, 0.4, 0.2],    # Marron clair
            12: [0.1, 0.6, 0.2],    # Vert
            13: [0.3, 0.6, 0.8],    # Bleu clair
            14: [0.4, 0.1, 0.6]     # Violet foncé
        }
        
        # Paramètres de validation
        self.validation_enabled = True
        self.min_distance_between_signs = 25
        self.confidence_threshold = 0.7
        
    def preprocess_image(self, image):
        """
        Prétraitement optimisé de l'image
        """
        # Amélioration du contraste local
        image_enhanced = ski.exposure.equalize_adapthist(
            image, 
            clip_limit=0.02, 
            kernel_size=None
        )
        
        # Filtrage gaussien léger pour réduire le bruit
        image_filtered = ski.filters.gaussian(image_enhanced, sigma=0.8)
        
        # Conversion HSV pour l'analyse couleur
        hsv = ski.color.rgb2hsv(image_filtered)
        
        return image_filtered, hsv
    
    def detect_circular_regions_simplified(self, image):
        """
        Détection simplifiée et plus précise des signes de métro
        Focus sur les vraies caractéristiques des panneaux
        """
        candidates = []
        
        # Conversion en niveaux de gris
        gray = ski.color.rgb2gray(image)
        
        # Détection de cercles avec Hough Transform - plus fiable pour les formes circulaires
        try:
            # Conversion pour OpenCV
            gray_uint8 = (gray * 255).astype(np.uint8)
            
            # Détection de cercles avec paramètres ajustés pour les signes de métro
            circles = cv2.HoughCircles(
                gray_uint8,
                cv2.HOUGH_GRADIENT,
                dp=1,                    # Résolution
                minDist=30,              # Distance minimale entre cercles
                param1=50,               # Seuil Canny élevé
                param2=30,               # Seuil d'accumulation
                minRadius=15,            # Rayon minimum
                maxRadius=60             # Rayon maximum
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # Vérifier les limites
                    x1 = max(0, x - r - 5)
                    x2 = min(image.shape[1], x + r + 5)
                    y1 = max(0, y - r - 5)
                    y2 = min(image.shape[0], y + r + 5)
                    
                    # Vérifier que c'est une région valide
                    if x2 > x1 and y2 > y1:
                        # Extraire la région pour analyse couleur
                        region = image[y1:y2, x1:x2]
                        
                        # Vérifier si la région contient des couleurs typiques des lignes
                        if self._has_metro_line_colors(region):
                            candidates.append({
                                'bbox': (x1, x2, y1, y2),
                                'area': np.pi * r * r,
                                'circularity': 1.0,
                                'centroid': (y, x),
                                'method': 'hough_targeted'
                            })
        
        except Exception as e:
            print(f"Erreur Hough circles: {e}")
        
        # Méthode de backup : détection par couleur ciblée
        candidates.extend(self._detect_by_specific_colors(image))
        
        return candidates

    def _has_metro_line_colors(self, region):
        """
        Vérifie si une région contient des couleurs typiques des lignes de métro
        """
        if region.size == 0:
            return False
        
        # Calculer la couleur moyenne de la région
        mean_color = np.mean(region.reshape(-1, 3), axis=0)
        
        # Vérifier si proche d'une couleur de ligne connue
        for ligne, ref_color in self.ligne_colors.items():
            ref_color = np.array(ref_color)
            distance = np.linalg.norm(mean_color - ref_color)
            
            if distance < 0.4:  # Seuil de tolérance couleur
                return True
        
        return False

    def _detect_by_specific_colors(self, image):
        """
        Détection ciblée par couleurs spécifiques des lignes
        """
        candidates = []
        
        # Conversion HSV pour meilleure segmentation
        hsv = ski.color.rgb2hsv(image)
        
        # Couleurs principales des lignes (HSV approximatif)
        target_colors_hsv = [
            # [H_min, H_max, S_min, V_min] pour chaque groupe de couleurs
            [0.0, 0.1, 0.4, 0.4],     # Rouge/Orange (lignes 5, 7)
            [0.1, 0.2, 0.4, 0.4],     # Jaune (ligne 1)
            [0.55, 0.75, 0.4, 0.4],   # Bleu (lignes 2, 8, 13)
            [0.25, 0.45, 0.4, 0.4],   # Vert (lignes 3, 6, 9, 12)
            [0.75, 0.95, 0.4, 0.4],   # Violet (lignes 4, 14)
            [0.95, 1.0, 0.4, 0.4],    # Rose/Magenta
        ]
        
        for h_min, h_max, s_min, v_min in target_colors_hsv:
            # Créer le masque pour cette couleur
            mask = (
                (hsv[:, :, 0] >= h_min) & (hsv[:, :, 0] <= h_max) &
                (hsv[:, :, 1] >= s_min) &
                (hsv[:, :, 2] >= v_min)
            )
            
            # Nettoyage morphologique
            mask_cleaned = morphology.opening(mask, morphology.disk(3))
            mask_cleaned = morphology.closing(mask_cleaned, morphology.disk(8))
            
            # Étiquetage des composantes connexes
            labeled = measure.label(mask_cleaned)
            
            for region in measure.regionprops(labeled):
                # Filtres très stricts pour les signes de métro
                if self._is_strict_metro_sign(region):
                    minr, minc, maxr, maxc = region.bbox
                    candidates.append({
                        'bbox': (minc, maxc, minr, maxr),
                        'area': region.area,
                        'circularity': self._calculate_circularity(region),
                        'centroid': region.centroid,
                        'method': 'color_targeted'
                    })
        
        return candidates

    def _is_strict_metro_sign(self, region):
        """
        Validation très stricte pour les signes de métro
        """
        area = region.area
        
        # Taille très spécifique aux signes de métro
        if area < 1000 or area > 5000:
            return False
        
        # Circularité très stricte
        circularity = self._calculate_circularity(region)
        if circularity < 0.7:  # Très rond obligatoire
            return False
        
        # Aspect ratio très strict
        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc
        height = maxr - minr
        
        if width == 0 or height == 0:
            return False
        
        aspect_ratio = min(width/height, height/width)
        if aspect_ratio < 0.85:  # Quasi carré obligatoire
            return False
        
        # Taille absolue minimale
        if width < 30 or height < 30:
            return False
        
        # Taille absolue maximale
        if width > 80 or height > 80:
            return False
        
        return True
    
    def _detect_by_contours(self, image):
        """
        Détection basée sur l'analyse de contours
        """
        # Conversion en niveaux de gris
        gray = ski.color.rgb2gray(image)
        
        # Détection de contours multi-échelle
        candidates = []
        
        for sigma in [0.8, 1.2, 1.6]:
            # Détection de contours avec différents niveaux de lissage
            edges = ski.feature.canny(
                gray, 
                sigma=sigma, 
                low_threshold=0.08, 
                high_threshold=0.15
            )
            
            # Fermeture morphologique
            kernel = morphology.disk(2)
            edges_closed = morphology.closing(edges, kernel)
            
            # Remplissage des trous
            filled = ndimage.binary_fill_holes(edges_closed)
            
            # Nettoyage des petits objets
            cleaned = morphology.remove_small_objects(filled, min_size=50)
            
            # Étiquetage et analyse des régions
            labeled = measure.label(cleaned)
            
            for region in measure.regionprops(labeled):
                if self._is_valid_sign_region(region):
                    if region.area > 0:
                        equivalent_diameter = np.sqrt(4 * region.area / np.pi)
                        bbox_area = (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
                        
                        # Ratio aire/bbox doit être élevé pour un cercle
                        if region.area / bbox_area > 0.65:
                            minr, minc, maxr, maxc = region.bbox
                            candidates.append({
                                'bbox': (minc, maxc, minr, maxr),
                                'area': region.area,
                                'circularity': self._calculate_circularity(region),
                                'centroid': region.centroid,
                                'method': 'contours'
                            })
        
        return candidates
    
    def _detect_by_color_segmentation(self, image):
        """
        Détection basée sur la segmentation couleur
        """
        candidates = []
        
        # Conversion HSV
        hsv = ski.color.rgb2hsv(image)
        
        # Pour chaque couleur de ligne, créer un masque
        for ligne, rgb_color in self.ligne_colors.items():
            # Conversion en HSV
            hsv_color = ski.color.rgb2hsv(np.array([[rgb_color]]))[0, 0]
            
            # Création du masque couleur avec tolérance
            h_tolerance = 0.1
            s_min, v_min = 0.3, 0.3
            
            mask = (
                (np.abs(hsv[:, :, 0] - hsv_color[0]) < h_tolerance) &
                (hsv[:, :, 1] > s_min) &
                (hsv[:, :, 2] > v_min)
            )
            
            # Nettoyage morphologique du masque
            mask_cleaned = morphology.opening(mask, morphology.disk(3))
            mask_cleaned = morphology.closing(mask_cleaned, morphology.disk(5))
            
            # Analyse des composantes connexes
            labeled = measure.label(mask_cleaned)
            
            for region in measure.regionprops(labeled):
                if self._is_valid_sign_region(region):
                    minr, minc, maxr, maxc = region.bbox
                    candidates.append({
                        'bbox': (minc, maxc, minr, maxr),
                        'area': region.area,
                        'circularity': self._calculate_circularity(region),
                        'centroid': region.centroid,
                        'method': 'color',
                        'predicted_line': ligne
                    })
        
        return candidates
    
    def _detect_by_hough_circles(self, image):
        """
        Détection par transformation de Hough (nécessite cv2)
        """
        candidates = []
        
        # Conversion en niveaux de gris pour OpenCV
        gray = ski.color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        # Détection de cercles
        circles = cv2.HoughCircles(
            gray_uint8,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=40,
            param2=25,
            minRadius=12,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Vérification des limites
                x1 = max(0, x - r)
                x2 = min(image.shape[1], x + r)
                y1 = max(0, y - r)
                y2 = min(image.shape[0], y + r)
                
                if x2 > x1 and y2 > y1:
                    candidates.append({
                        'bbox': (x1, x2, y1, y2),
                        'area': np.pi * r * r,
                        'circularity': 1.0,
                        'centroid': (y, x),
                        'method': 'hough'
                    })
        
        return candidates
    
    def _is_valid_sign_region(self, region):
        area = region.area
        
        # Filtrage par taille (plus restrictif)
        if area < 800 or area > 4000:  # Plus strict
            return False
        
        # Filtrage par circularité (plus strict)
        circularity = self._calculate_circularity(region)
        if circularity < 0.6:  # Plus strict (était 0.3)
            return False
        
        # Filtrage par rapport largeur/hauteur (plus strict)
        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc
        height = maxr - minr
        
        if width == 0 or height == 0:
            return False
        
        aspect_ratio = min(width/height, height/width)
        if aspect_ratio < 0.8:  # Plus strict (était 0.6)
            return False
        
        # Nouveau filtre : taille minimale absolue
        if width < 25 or height < 25:
            return False
            
        return True
    
    def _calculate_circularity(self, region):
        """
        Calcule la circularité d'une région
        """
        area = region.area
        perimeter = region.perimeter
        
        if perimeter == 0:
            return 0
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity
    
    def extract_color_features_advanced(self, image, bbox):
        """
        Extraction avancée des caractéristiques de couleur
        """
        x1, x2, y1, y2 = bbox
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        h, w = region.shape[:2]
        
        # Création de plusieurs masques pour différentes zones
        center = (w//2, h//2)
        
        # Zone centrale (noyau)
        radius_core = min(w, h) // 6
        y_coords, x_coords = np.ogrid[:h, :w]
        mask_core = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius_core**2
        
        # Zone intermédiaire
        radius_mid = min(w, h) // 4
        mask_mid = ((x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius_mid**2) & ~mask_core
        
        features = {}
        
        # Couleur du noyau (plus fiable)
        if np.any(mask_core):
            features['core_color'] = np.median(region[mask_core], axis=0)
            features['core_std'] = np.std(region[mask_core], axis=0)
        else:
            features['core_color'] = np.median(region.reshape(-1, 3), axis=0)
            features['core_std'] = np.std(region.reshape(-1, 3), axis=0)
        
        # Couleur intermédiaire
        if np.any(mask_mid):
            features['mid_color'] = np.median(region[mask_mid], axis=0)
        else:
            features['mid_color'] = features['core_color']
        
        # Histogramme de couleurs dominant
        hist_r = np.histogram(region[:, :, 0].flatten(), bins=32, range=(0, 1))[0]
        hist_g = np.histogram(region[:, :, 1].flatten(), bins=32, range=(0, 1))[0]
        hist_b = np.histogram(region[:, :, 2].flatten(), bins=32, range=(0, 1))[0]
        
        # Pic dominant pour chaque canal
        features['dominant_r'] = np.argmax(hist_r) / 32.0
        features['dominant_g'] = np.argmax(hist_g) / 32.0
        features['dominant_b'] = np.argmax(hist_b) / 32.0
        
        return features
    
    def classify_line_advanced(self, image, bbox, candidate_info=None):
        """
        Classification avancée du numéro de ligne
        """
        # Extraction des caractéristiques
        features = self.extract_color_features_advanced(image, bbox)
        
        if features is None:
            return 1
        
        # Utiliser prioritairement la couleur du noyau
        main_color = features['core_color']
        
        # Si une prédiction couleur existe déjà (segmentation couleur)
        if candidate_info and 'predicted_line' in candidate_info:
            # Vérifier la cohérence
            predicted_line = candidate_info['predicted_line']
            color_distance = np.linalg.norm(
                main_color - np.array(self.ligne_colors[predicted_line])
            )
            
            if color_distance < 0.3:  # Seuil de cohérence
                return predicted_line
        
        # Classification par distance euclidienne avec pondération
        best_match = 1
        min_weighted_distance = float('inf')
        
        for ligne, ref_color in self.ligne_colors.items():
            ref_color = np.array(ref_color)
            
            # Distance principale (couleur noyau)
            core_distance = np.linalg.norm(main_color - ref_color)
            
            # Distance couleurs dominantes
            dominant_color = np.array([
                features['dominant_r'],
                features['dominant_g'], 
                features['dominant_b']
            ])
            dominant_distance = np.linalg.norm(dominant_color - ref_color)
            
            # Distance pondérée
            weighted_distance = 0.7 * core_distance + 0.3 * dominant_distance
            
            if weighted_distance < min_weighted_distance:
                min_weighted_distance = weighted_distance
                best_match = ligne
        
        return best_match
    
    def remove_overlapping_detections(self, candidates):
        """
        Supprime les détections qui se chevauchent
        """
        if len(candidates) <= 1:
            return candidates
        
        # Calcul des IoU (Intersection over Union)
        filtered_candidates = []
        
        # Trier par confiance/aire (plus grandes d'abord)
        candidates_sorted = sorted(candidates, 
                                 key=lambda x: x.get('area', 0), reverse=True)
        
        for i, candidate in enumerate(candidates_sorted):
            bbox1 = candidate['bbox']
            
            # Vérifier le chevauchement avec les candidats déjà acceptés
            overlaps = False
            
            for accepted in filtered_candidates:
                bbox2 = accepted['bbox']
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > 0.3:  # Seuil de chevauchement
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_candidates.append(candidate)
        
        return filtered_candidates
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calcule l'Intersection over Union entre deux boîtes
        """
        x1_1, x2_1, y1_1, y2_1 = bbox1
        x1_2, x2_2, y1_2, y2_2 = bbox2
        
        # Intersection
        x1_inter = max(x1_1, x1_2)
        x2_inter = min(x2_1, x2_2)
        y1_inter = max(y1_1, y1_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        area_union = area1 + area2 - area_inter
        
        if area_union == 0:
            return 0.0
        
        return area_inter / area_union
    
    def estimate_detection_confidence(self, image, candidate):
        """
        Estime la confiance d'une détection
        """
        bbox = candidate['bbox']
        x1, x2, y1, y2 = bbox
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        scores = []
        
        # Score de circularité
        circularity = candidate.get('circularity', 0.5)
        circularity_score = min(circularity / 0.8, 1.0)
        scores.append(circularity_score)
        
        # Score de taille
        area = candidate.get('area', 0)
        optimal_area = 2000  # Taille optimale approximative
        size_score = 1.0 - abs(area - optimal_area) / optimal_area
        size_score = max(0.0, min(1.0, size_score))
        scores.append(size_score)
        
        # Score de contraste
        if len(region.shape) == 3:
            gray = np.mean(region, axis=2)
        else:
            gray = region
        
        contrast = np.std(gray)
        contrast_score = min(contrast / 0.25, 1.0)
        scores.append(contrast_score)
        
        # Bonus pour certaines méthodes de détection
        method_bonus = {
            'contours': 0.0,
            'color': 0.1,
            'hough': 0.05
        }
        
        method = candidate.get('method', 'contours')
        bonus = method_bonus.get(method, 0.0)
        
        confidence = np.mean(scores) + bonus
        return min(1.0, confidence)


def processOneMetroImage(nom, im, n, resizeFactor):
    """
    Fonction principale de traitement - Version finale optimisée
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
    
    # Initialisation du détecteur
    detector = MetroLineDetector()
    
    # Prétraitement
    image_processed, hsv = detector.preprocess_image(im_resized)
    
    # Détection multi-méthodes
    candidates = detector.detect_circular_regions_simplified(image_processed)
    
    # Suppression des chevauchements
    candidates = detector.remove_overlapping_detections(candidates)
    
    # Filtrage par confiance
    confident_candidates = []
    for candidate in candidates:
        confidence = detector.estimate_detection_confidence(image_processed, candidate)
        if confidence >= detector.confidence_threshold:
            confident_candidates.append(candidate)
    
    # Limitation du nombre de détections
    if len(confident_candidates) > 8:
        # Trier par confiance et garder les 8 meilleures
        candidates_with_conf = [
            (candidate, detector.estimate_detection_confidence(image_processed, candidate))
            for candidate in confident_candidates
        ]
        candidates_with_conf.sort(key=lambda x: x[1], reverse=True)
        confident_candidates = [c[0] for c in candidates_with_conf[:8]]
    
    # Construction du résultat final
    bd = []
    
    for candidate in confident_candidates:
        x1, x2, y1, y2 = candidate['bbox']
        
        # Classification du numéro de ligne
        ligne = detector.classify_line_advanced(im_resized, (x1, x2, y1, y2), candidate)
        
        # Ajout au résultat
        bd.append([n, x1, x2, y1, y2, ligne])
    
    # Conversion en numpy array
    if bd:
        bd = np.array(bd)
    else:
        bd = np.empty((0, 6))
    
    # Affichage des résultats
    # plt.figure(figsize=(12, 8))
    # plt.imshow(im_resized)
    
    # if bd.size > 0:
    #     for k in range(bd.shape[0]):
    #         draw_rectangle(bd[k,3], bd[k,4], bd[k,1], bd[k,2], 'g')
    #         # Afficher le numéro de ligne détecté
    #         center_x = (bd[k,1] + bd[k,2]) / 2
    #         center_y = (bd[k,3] + bd[k,4]) / 2
    #         plt.text(center_x, center_y, str(int(bd[k,5])), 
    #                 color='red', fontsize=12, fontweight='bold',
    #                 ha='center', va='center')
        
    #     lignes_detectees = bd[:,5].astype(int)
    #     plt.title(f'{nom} - Lignes détectées: {lignes_detectees} ({len(lignes_detectees)} signes)')
    # else:
    #     plt.title(f'{nom} - Aucune ligne détectée')
    
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.imshow(im_resized)

    if bd.size > 0:
        for k in range(bd.shape[0]):
            draw_rectangle(bd[k,3], bd[k,4], bd[k,1], bd[k,2], 'g')
            center_x = (bd[k,1] + bd[k,2]) / 2
            center_y = (bd[k,3] + bd[k,4]) / 2
            plt.text(center_x, center_y, str(int(bd[k,5])), 
                    color='red', fontsize=12, fontweight='bold',
                    ha='center', va='center')
        
        lignes_detectees = bd[:,5].astype(int)
        plt.title(f'{nom} - Lignes détectées: {lignes_detectees} ({len(lignes_detectees)} signes)')
    else:
        plt.title(f'{nom} - Aucune ligne détectée')

    plt.axis('off')
    plt.tight_layout()

    # Sauvegarde automatique
    import os
    output_dir = 'results_images'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{nom}_detected.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Important : ferme automatiquement

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