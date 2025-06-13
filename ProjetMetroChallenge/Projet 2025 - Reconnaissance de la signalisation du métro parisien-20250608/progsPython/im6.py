# -*- coding: utf-8 -*-
"""
Système de détection de métro avec reconnaissance de chiffres
Combine détection de cercles + reconnaissance du contenu + validation couleur
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage as ski
from skimage import color, morphology, measure, feature
from skimage.transform import resize
import cv2
import scipy.io as sio
from PIL import Image
import os
from collections import defaultdict
from PIL import Image
import numpy as np

# Charger l'image
im = np.array(Image.open("../BD_METRO/IM (6).JPG").convert('RGB')) / 255.0


class DigitAwareMetroSystem:
    """
    Système qui combine détection de cercles, reconnaissance de chiffres et couleurs
    """
    
    def __init__(self):
        # Chargement des données d'apprentissage complètes
        self.learned_colors = {}
        self.is_trained = False
        
        # Templates de chiffres simplifiés (patterns attendus)
        self.expected_digits = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
        
        # Configurations HoughCircles équilibrées
        self.hough_configs = [
            {
                'name': 'balanced',
                'dp': 1, 'minDist': 60, 'param1': 50, 'param2': 40,
                'minRadius': 25, 'maxRadius': 55
            }
        ]
    
    def train_from_full_dataset(self, images_folder, gt_file, resize_factor=1.0):
        """
        Entraînement sur TOUTE la base d'apprentissage
        """
        print("🎓 ENTRAÎNEMENT SUR TOUTE LA BASE D'APPRENTISSAGE")
        print("="*60)
        
        gt_data = sio.loadmat(gt_file)['BD']
        samples_by_line = defaultdict(list)
        
        print(f"Total d'annotations: {len(gt_data)}")
        
        for annotation in gt_data:
            img_num, x1_orig, x2_orig, y1_orig, y2_orig, ligne = annotation
            img_num, ligne = int(img_num), int(ligne)
            
            # Corriger les coordonnées (inversion x/y)
            x1, x2 = int(y1_orig), int(y2_orig)
            y1, y2 = int(x1_orig), int(x2_orig)
            
            # Charger l'image
            img_path = os.path.join(images_folder, f'IM ({img_num}).JPG')
            if os.path.exists(img_path):
                try:
                    image = np.array(Image.open(img_path).convert('RGB')) / 255.0
                    
                    if resize_factor != 1.0:
                        image = resize(image, 
                                     (int(image.shape[0] * resize_factor), 
                                      int(image.shape[1] * resize_factor)),
                                     anti_aliasing=True, preserve_range=True).astype(image.dtype)
                        x1, x2, y1, y2 = (np.array([x1, x2, y1, y2]) * resize_factor).astype(int)
                    
                    # Analyser cette région
                    region = image[y1:y2, x1:x2]
                    h, w = region.shape[:2]
                    
                    if h > 20 and w > 20:  # Région valide
                        # Extraire couleur de manière robuste
                        extracted_color = self._extract_robust_color(region)
                        
                        samples_by_line[ligne].append({
                            'color': extracted_color,
                            'img_num': img_num,
                            'region_size': (w, h)
                        })
                        
                except Exception as e:
                    print(f"Erreur image {img_num}: {e}")
                    continue
        
        # Apprendre les couleurs pour chaque ligne
        print(f"\n📊 COULEURS APPRISES:")
        for ligne in sorted(samples_by_line.keys()):
            samples = samples_by_line[ligne]
            if len(samples) >= 3:  # Au moins 3 échantillons
                colors = np.array([sample['color'] for sample in samples])
                
                # Utiliser la médiane pour être robuste aux outliers
                learned_color = np.median(colors, axis=0)
                self.learned_colors[ligne] = learned_color
                
                print(f"   L{ligne}: {len(samples)} échantillons → {learned_color}")
        
        self.is_trained = True
        print(f"\n✅ Apprentissage terminé sur {len(self.learned_colors)} lignes")
        return True
    
    def detect_metro_signs(self, image, debug=True):
        """
        Détection avec validation par chiffre ET couleur
        """
        if not self.is_trained:
            print("⚠️ Système non entraîné !")
            return []
        
        if debug:
            print("🔍 Détection avec RECONNAISSANCE DE CHIFFRES...")
        
        # Étape 1: Détecter tous les cercles
        all_circles = []
        for config in self.hough_configs:
            circles = self._detect_circles_with_config(image, config, debug)
            for circle in circles:
                all_circles.append((*circle, config['name']))
        
        if not all_circles:
            return []
        
        # Étape 2: Fusionner cercles similaires
        unique_circles = self._merge_similar_circles(all_circles, debug)
        
        if debug:
            print(f"   Cercles candidats: {len(unique_circles)}")
        
        # Étape 3: FILTRER par contenu (chiffres vs lettres)
        metro_circles = self._filter_by_content(image, unique_circles, debug)
        
        if debug:
            print(f"   Cercles avec chiffres: {len(metro_circles)}")
        
        # Étape 4: Validation finale par couleur
        valid_detections = []
        for i, (x, y, r, config_name, detected_digit) in enumerate(metro_circles):
            if debug:
                print(f"\n   Validation L{detected_digit}: ({x}, {y}, r={r})")
            
            validation = self._validate_digit_and_color(image, x, y, r, detected_digit, debug)
            
            if validation['is_valid']:
                detection = {
                    'bbox': validation['bbox'],
                    'center': (x, y),
                    'radius': r,
                    'ligne': validation['ligne'],
                    'confidence': validation['confidence'],
                    'config_used': config_name,
                    'detected_digit': detected_digit,
                    'color_match': validation['color_match']
                }
                valid_detections.append(detection)
                
                if debug:
                    print(f"     ✅ L{validation['ligne']} confirmée "
                          f"(chiffre: {detected_digit}, conf: {validation['confidence']:.2f})")
            else:
                if debug:
                    print(f"     ❌ L{detected_digit} rejetée (conf: {validation['confidence']:.2f})")
        
        # Étape 5: Supprimer doublons
        final_detections = self._remove_duplicates(valid_detections, debug)
        
        if debug:
            print(f"\n🎯 DÉTECTIONS FINALES: {len(final_detections)}")
        
        return final_detections
    
    def _detect_circles_with_config(self, image, config, debug=False):
        """
        Détection de cercles standard
        """
        gray = color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        gray_uint8 = cv2.medianBlur(gray_uint8, 5)
        
        circles = cv2.HoughCircles(
            gray_uint8, cv2.HOUGH_GRADIENT,
            **{k:v for k,v in config.items() if k != 'name'}
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Filtrage des bords
            h, w = image.shape[:2]
            margin = 30
            valid_circles = []
            
            for x, y, r in circles:
                if (margin < x < w-margin and margin < y < h-margin):
                    valid_circles.append((x, y, r))
            
            if debug:
                print(f"     Cercles bruts: {len(circles)}, valides: {len(valid_circles)}")
            
            return valid_circles
        
        return []
    
    def _merge_similar_circles(self, all_circles, debug=False):
        """
        Fusion des cercles similaires
        """
        if not all_circles:
            return []
        
        merged = []
        used = set()
        
        for i, (x1, y1, r1, config1) in enumerate(all_circles):
            if i in used:
                continue
            
            similar_group = [(x1, y1, r1, config1)]
            used.add(i)
            
            for j, (x2, y2, r2, config2) in enumerate(all_circles[i+1:], i+1):
                if j in used:
                    continue
                
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < 20:
                    similar_group.append((x2, y2, r2, config2))
                    used.add(j)
            
            if len(similar_group) > 1:
                xs = [c[0] for c in similar_group]
                ys = [c[1] for c in similar_group]
                rs = [c[2] for c in similar_group]
                
                median_x = int(np.median(xs))
                median_y = int(np.median(ys))
                median_r = int(np.median(rs))
                
                merged.append((median_x, median_y, median_r, similar_group[0][3]))
            else:
                merged.append(similar_group[0])
        
        return merged
    
    def _filter_by_content(self, image, circles, debug=False):
        """
        FILTRE CRUCIAL: Ne garder que les cercles avec des CHIFFRES
        """
        metro_circles = []
        
        for x, y, r, config_name in circles:
            # Extraire la zone centrale (où devrait être le chiffre)
            center_analysis = self._analyze_center_content(image, x, y, r)
            
            if debug:
                print(f"   Cercle ({x}, {y}): {center_analysis['content_type']} "
                      f"(chiffre détecté: {center_analysis.get('detected_digit', 'aucun')})")
            
            # Ne garder que si c'est un chiffre de métro valide
            if (center_analysis['content_type'] == 'digit' and 
                center_analysis.get('detected_digit') in self.expected_digits):
                
                metro_circles.append((x, y, r, config_name, center_analysis['detected_digit']))
        
        return metro_circles
    
    def _analyze_center_content(self, image, x, y, r):
        """
        Analyse le contenu du centre du cercle pour détecter chiffres vs lettres
        """
        # Extraire la région centrale
        h, w = image.shape[:2]
        margin = 5
        x1, x2 = max(0, x-r-margin), min(w, x+r+margin)
        y1, y2 = max(0, y-r-margin), min(h, y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return {'content_type': 'invalid'}
        
        region = image[y1:y2, x1:x2]
        region_h, region_w = region.shape[:2]
        
        # Zone centrale (où est le chiffre/lettre)
        center_x, center_y = region_w // 2, region_h // 2
        center_radius = min(region_w, region_h) // 3
        
        center_x1 = max(0, center_x - center_radius)
        center_x2 = min(region_w, center_x + center_radius)
        center_y1 = max(0, center_y - center_radius)
        center_y2 = min(region_h, center_y + center_radius)
        
        center_region = region[center_y1:center_y2, center_x1:center_x2]
        
        if center_region.size == 0:
            return {'content_type': 'invalid'}
        
        # Convertir en niveaux de gris et binariser
        gray_center = color.rgb2gray(center_region)
        
        # Le chiffre doit être plus clair que le fond
        threshold = np.mean(gray_center) + 0.1
        binary_center = gray_center > threshold
        
        # Analyser les caractéristiques
        white_ratio = np.sum(binary_center) / binary_center.size
        
        # Détection basique par analyse de forme
        if white_ratio < 0.1:  # Pas assez de pixels blancs
            return {'content_type': 'none'}
        
        if white_ratio > 0.8:  # Trop de blanc (probablement pas un chiffre)
            return {'content_type': 'background'}
        
        # Analyser la connectivité des composants blancs
        labeled = measure.label(binary_center)
        regions = measure.regionprops(labeled)
        
        if not regions:
            return {'content_type': 'none'}
        
        # Caractéristiques du plus grand composant
        largest_region = max(regions, key=lambda r: r.area)
        
        # Heuristiques pour distinguer chiffres vs lettres
        area_ratio = largest_region.area / binary_center.size
        eccentricity = largest_region.eccentricity
        extent = largest_region.extent
        
        # Détection très basique du chiffre par analyse géométrique
        detected_digit = self._estimate_digit_from_shape(binary_center, largest_region)
        
        # Classification simple : si ça ressemble à un chiffre de métro
        if (0.1 < area_ratio < 0.7 and 
            detected_digit in self.expected_digits):
            return {
                'content_type': 'digit',
                'detected_digit': detected_digit,
                'confidence': min(1.0, area_ratio * 2),
                'white_ratio': white_ratio
            }
        else:
            return {
                'content_type': 'letter_or_symbol',
                'white_ratio': white_ratio
            }
    
    def _estimate_digit_from_shape(self, binary_image, region):
        """
        Estimation très basique du chiffre par analyse géométrique
        """
        # Cette fonction est volontairement simple
        # Dans un vrai système, on utiliserait OCR ou template matching
        
        area_ratio = region.area / binary_image.size
        bbox = region.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 1
        
        # Heuristiques très basiques basées sur les observations
        if area_ratio < 0.15:  # Petit chiffre fin
            if aspect_ratio < 0.6:
                return 1  # Probablement 1 (fin et vertical)
            else:
                return 7  # Probablement 7
        elif area_ratio < 0.25:  # Chiffre moyen
            if aspect_ratio > 0.8:
                return 8  # Probablement 8 (carré)
            else:
                return 4  # Probablement 4
        elif area_ratio < 0.35:  # Chiffre plus large
            if region.eccentricity > 0.7:
                return 11  # Probablement 11 (allongé)
            else:
                return 12  # Probablement 12
        else:  # Grand chiffre
            if aspect_ratio > 0.9:
                return 14  # Probablement 14
            else:
                return 13  # Probablement 13
        
        # Par défaut, deviner selon la position dans l'image
        # (les lignes 1, 4, 7, 11, 14 sont plus fréquentes)
        return np.random.choice([1, 4, 7, 11, 14])
    
    def _validate_digit_and_color(self, image, x, y, r, detected_digit, debug=False):
        """
        Validation finale : le chiffre détecté correspond-il à la couleur ?
        """
        # Extraire région pour analyse couleur
        h, w = image.shape[:2]
        margin = 8
        x1, x2 = max(0, x-r-margin), min(w, x+r+margin)
        y1, y2 = max(0, y-r-margin), min(h, y+r+margin)
        
        if x2 <= x1 or y2 <= y1:
            return {'is_valid': False, 'confidence': 0.0}
        
        region = image[y1:y2, x1:x2]
        extracted_color = self._extract_robust_color(region)
        
        # Vérifier si on a une couleur apprise pour ce chiffre
        if detected_digit not in self.learned_colors:
            if debug:
                print(f"       Ligne {detected_digit} non apprise")
            return {'is_valid': False, 'confidence': 0.0}
        
        # Comparer avec la couleur apprise
        learned_color = self.learned_colors[detected_digit]
        distance = np.linalg.norm(extracted_color - learned_color)
        
        # Tolérance adaptative (plus stricte)
        tolerance = 0.3
        confidence = max(0, 1.0 - distance / tolerance)
        
        # Validation : chiffre ET couleur doivent correspondre
        is_valid = confidence > 0.6  # Seuil strict
        
        if debug:
            print(f"       Couleur extraite: {extracted_color}")
            print(f"       Couleur apprise L{detected_digit}: {learned_color}")
            print(f"       Distance: {distance:.3f}, Confidence: {confidence:.3f}")
        
        return {
            'is_valid': is_valid,
            'ligne': detected_digit if is_valid else -1,
            'confidence': confidence,
            'color_match': 1.0 - distance,
            'bbox': (x1, x2, y1, y2) if is_valid else None
        }
    
    def _extract_robust_color(self, region):
        """
        Extraction robuste de couleur (zone périphérique)
        """
        h, w = region.shape[:2]
        center_y, center_x = h//2, w//2
        
        # Créer masque pour zone périphérique
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        radius = min(h, w) // 2
        ring_mask = distances >= radius * 0.6
        
        if np.any(ring_mask):
            ring_pixels = region[ring_mask]
            return np.median(ring_pixels, axis=0)
        else:
            return np.mean(region.reshape(-1, 3), axis=0)
    
    def _remove_duplicates(self, detections, debug=False):
        """
        Supprime les détections dupliquées
        """
        if not detections:
            return detections
        
        final = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            duplicates = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                dist = np.sqrt((det1['center'][0] - det2['center'][0])**2 + 
                             (det1['center'][1] - det2['center'][1])**2)
                
                if dist < 30:
                    duplicates.append(det2)
                    used.add(j)
            
            best = max(duplicates, key=lambda d: d['confidence'])
            final.append(best)
        
        return final


def processOneMetroImageDigitAware(nom, im, n, resizeFactor, metro_system, save_images=False):
    """
    Fonction avec reconnaissance de chiffres
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
    print(f"Traitement avec RECONNAISSANCE CHIFFRES: {nom}")
    print(f"{'='*60}")
    
    # Lancer la détection
    detections = metro_system.detect_metro_signs(im_resized, debug=True)
    
    # Convertir au format attendu
    bd = []
    for detection in detections:
        if detection['bbox']:
            x1, x2, y1, y2 = detection['bbox']
            bd.append([n, x1, x2, y1, y2, detection['ligne']])
    
    bd = np.array(bd) if bd else np.empty((0, 6))
    
    # Affichage avec info chiffres détectés
    plt.figure(figsize=(16, 10))
    plt.imshow(im_resized)
    
    colors = ['lime', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
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
            
            plt.text(x1, y1-35, f"CHIFFRE: {detection['detected_digit']}", 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", 
                            facecolor='darkgreen', alpha=0.8))
            
            plt.text(x2, y2+20, f"Conf: {detection['confidence']:.2f}", 
                    color='white', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", 
                            facecolor='black', alpha=0.7))
    
    titre = f"{nom} - {len(detections)} détection(s) avec CHIFFRES"
    if detections:
        lignes = [str(d['ligne']) for d in detections]
        chiffres = [str(d['detected_digit']) for d in detections]
        titre += f"\nLignes: {', '.join(lignes)} | Chiffres: {', '.join(chiffres)}"
    
    plt.title(titre, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\n🎯 RÉSULTAT avec CHIFFRES: {len(detections)} signe(s)")
    for det in detections:
        print(f"  ✅ Ligne {det['ligne']} (chiffre: {det['detected_digit']}, "
              f"conf: {det['confidence']:.3f})")
    
    return im_resized, bd


if __name__ == "__main__":
    print("Système Metro avec reconnaissance de chiffres prêt.")
    print("Usage:")
    print("1. system = DigitAwareMetroSystem()")
    print("2. system.train_from_full_dataset('../BD_METRO', 'Apprentissage.mat')")
    print("3. Utiliser processOneMetroImageDigitAware()")
    # Créer et entraîner le système sur TOUTE la base
    system = DigitAwareMetroSystem()
    system.train_from_full_dataset('../BD_METRO', 'Apprentissage.mat')

    # Tester sur l'image 6
    im_resized, bd = processOneMetroImageDigitAware("IM (6)", im, 6, 1.0, system)