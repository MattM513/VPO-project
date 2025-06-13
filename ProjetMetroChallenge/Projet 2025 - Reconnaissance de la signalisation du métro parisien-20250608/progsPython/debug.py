# -*- coding: utf-8 -*-
"""
Programme de debug pour la détection de signes de métro
Visualise chaque étape du processus
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import skimage as ski
from skimage import color
import cv2
from PIL import Image


class MetroDebugger:
    """
    Classe pour debugger la détection de signes de métro
    """
    
    def __init__(self):
        # Couleurs de référence (RGB 0-1)
        self.ligne_colors = {
            1: [1.0, 0.808, 0.0],          # #FFCE00 - Jaune
            2: [0.0, 0.392, 0.690],        # #0064B0 - Bleu
            3: [0.624, 0.596, 0.145],      # #9F9825 - Vert olive
            4: [0.753, 0.255, 0.569],      # #C04191 - Rose/Magenta
            5: [0.949, 0.557, 0.259],      # #F28E42 - Orange
            6: [0.514, 0.769, 0.569],      # #83C491 - Vert clair
            7: [0.953, 0.643, 0.729],      # #F3A4BA - Rose clair
            8: [0.808, 0.678, 0.824],      # #CEADD2 - Mauve
            9: [0.835, 0.788, 0.0],        # #D5C900 - Jaune-vert
            10: [0.890, 0.702, 0.165],     # #E3B32A - Jaune orangé
            11: [0.553, 0.369, 0.165],     # #8D5E2A - Marron
            12: [0.0, 0.506, 0.310],       # #00814F - Vert
            13: [0.596, 0.831, 0.886],     # #98D4E2 - Bleu clair
            14: [0.400, 0.141, 0.514]      # #662483 - Violet
        }
    
    def debug_full_pipeline(self, image_path, max_circles_to_show=10):
        """
        Debug complet du pipeline de détection
        """
        print(f"🔍 DEBUT DEBUG sur {image_path}")
        print("="*60)
        
        # Charger l'image
        if isinstance(image_path, str):
            image = np.array(Image.open(image_path).convert('RGB')) / 255.0
        else:
            image = image_path  # Si c'est déjà un array numpy
        
        print(f"📐 Taille image: {image.shape}")
        
        # 1. Afficher l'image originale
        plt.figure(figsize=(20, 15))
        
        # Image originale
        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title("Image originale", fontsize=14)
        plt.axis('off')
        
        # 2. Debug détection de cercles
        circles = self._debug_circle_detection(image)
        
        # 3. Afficher tous les cercles détectés
        plt.subplot(2, 3, 2)
        plt.imshow(image)
        plt.title(f"Tous les cercles détectés ({len(circles)})", fontsize=14)
        
        for i, (x, y, r) in enumerate(circles[:50]):  # Max 50 pour la lisibilité
            circle = Circle((x, y), r, fill=False, color='red', linewidth=1, alpha=0.7)
            plt.gca().add_patch(circle)
            if i < 10:  # Numéroter les 10 premiers
                plt.text(x, y, str(i+1), color='white', fontsize=8, 
                        ha='center', va='center', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.8))
        plt.axis('off')
        
        # 4. Analyser quelques cercles en détail
        plt.subplot(2, 3, 3)
        self._debug_color_analysis(image, circles[:max_circles_to_show])
        
        # 5. Tester différents paramètres HoughCircles
        plt.subplot(2, 3, 4)
        self._debug_hough_parameters(image)
        
        # 6. Analyser les couleurs de l'image
        plt.subplot(2, 3, 5)
        self._debug_color_distribution(image)
        
        # 7. Test avec segmentation couleur
        plt.subplot(2, 3, 6)
        self._debug_color_segmentation(image)
        
        plt.tight_layout()
        plt.show()
        
        # 8. Analyse détaillée des meilleurs candidats
        if circles:
            self._detailed_circle_analysis(image, circles[:max_circles_to_show])
        
        return circles
    
    def _debug_circle_detection(self, image):
        """
        Debug de la détection de cercles avec différents paramètres
        """
        print("\n🔵 DEBUG DETECTION CERCLES")
        print("-" * 30)
        
        gray = color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        gray_uint8 = cv2.medianBlur(gray_uint8, 5)
        
        # Test avec paramètres actuels
        circles1 = cv2.HoughCircles(
            gray_uint8, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=20, maxRadius=60
        )
        
        # Test avec paramètres plus permissifs
        circles2 = cv2.HoughCircles(
            gray_uint8, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=30, param2=20, minRadius=15, maxRadius=80
        )
        
        # Test avec paramètres plus stricts
        circles3 = cv2.HoughCircles(
            gray_uint8, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
            param1=70, param2=40, minRadius=25, maxRadius=50
        )
        
        results = []
        if circles1 is not None:
            results.extend(np.round(circles1[0, :]).astype("int").tolist())
        
        print(f"Paramètres actuels: {len(results) if circles1 is not None else 0} cercles")
        print(f"Paramètres permissifs: {len(circles2[0]) if circles2 is not None else 0} cercles")
        print(f"Paramètres stricts: {len(circles3[0]) if circles3 is not None else 0} cercles")
        
        return results
    
    def _debug_color_analysis(self, image, circles):
        """
        Affiche les couleurs extraites pour chaque cercle
        """
        plt.imshow(np.ones((100, 100, 3)))  # Fond blanc
        plt.title("Couleurs extraites", fontsize=14)
        
        colors_found = []
        
        for i, (x, y, r) in enumerate(circles[:5]):  # Max 5 cercles
            # Extraire la région
            region = self._extract_circle_region(image, x, y, r)
            if region is not None:
                dominant_color = self._get_dominant_color(region, r)
                colors_found.append(dominant_color)
                
                # Afficher la couleur
                rect = Rectangle((i*15, 50), 10, 10, 
                               facecolor=dominant_color, edgecolor='black')
                plt.gca().add_patch(rect)
                plt.text(i*15+5, 40, f"C{i+1}", ha='center', fontsize=8)
        
        # Afficher les couleurs de référence
        for j, (ligne, ref_color) in enumerate(list(self.ligne_colors.items())[:7]):
            rect = Rectangle((j*12, 20), 8, 8, 
                           facecolor=ref_color, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(j*12+4, 10, f"L{ligne}", ha='center', fontsize=8)
        
        plt.xlim(0, 100)
        plt.ylim(0, 80)
        plt.axis('off')
    
    def _debug_hough_parameters(self, image):
        """
        Test différents paramètres Hough
        """
        gray = color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        gray_uint8 = cv2.medianBlur(gray_uint8, 5)
        
        # Afficher l'image préprocessée
        plt.imshow(gray_uint8, cmap='gray')
        plt.title("Image préprocessée (niveaux de gris)", fontsize=14)
        plt.axis('off')
    
    def _debug_color_distribution(self, image):
        """
        Analyse la distribution des couleurs dans l'image
        """
        # Calculer l'histogramme des couleurs
        pixels = image.reshape(-1, 3)
        
        plt.hist2d(pixels[:, 0], pixels[:, 1], bins=50, alpha=0.6)
        plt.xlabel('Rouge')
        plt.ylabel('Vert')
        plt.title('Distribution Rouge-Vert', fontsize=14)
        
        # Marquer les couleurs de référence
        for ligne, color in list(self.ligne_colors.items())[:5]:
            plt.scatter(color[0], color[1], c=[color], s=100, 
                       edgecolor='black', linewidth=2, label=f'L{ligne}')
        
        plt.legend(fontsize=8)
    
    def _debug_color_segmentation(self, image):
        """
        Test de segmentation par couleur pour la ligne 4 (comme dans votre code original)
        """
        # Couleur ligne 4 (rose/magenta)
        ligne_4_color = np.array([0.753, 0.255, 0.569])
        tolerance = 0.15
        
        # Calculer les distances
        distances = np.linalg.norm(image - ligne_4_color, axis=2)
        mask = distances < tolerance
        
        # Afficher le masque
        plt.imshow(mask, cmap='hot')
        plt.title(f'Segmentation Ligne 4 (tolerance={tolerance})', fontsize=14)
        plt.axis('off')
        
        print(f"Pixels détectés pour ligne 4: {np.sum(mask)}")
    
    def _detailed_circle_analysis(self, image, circles):
        """
        Analyse détaillée des premiers cercles
        """
        print(f"\n🔍 ANALYSE DETAILLEE des {min(len(circles), 5)} premiers cercles")
        print("="*60)
        
        fig, axes = plt.subplots(2, min(len(circles), 5), figsize=(20, 8))
        if len(circles) == 1:
            axes = axes.reshape(2, 1)
        
        for i, (x, y, r) in enumerate(circles[:5]):
            print(f"\n--- CERCLE {i+1} ---")
            print(f"Position: ({x}, {y}), Rayon: {r}")
            
            # Extraire la région
            region = self._extract_circle_region(image, x, y, r)
            
            if region is not None:
                # Afficher la région originale
                axes[0, i].imshow(region)
                axes[0, i].set_title(f'Région {i+1}')
                axes[0, i].axis('off')
                
                # Analyser la couleur
                dominant_color = self._get_dominant_color(region, r)
                print(f"Couleur dominante: RGB{dominant_color}")
                
                # Trouver la ligne la plus proche
                best_line, distance = self._find_closest_line(dominant_color)
                print(f"Ligne la plus proche: {best_line} (distance: {distance:.3f})")
                
                # Analyser la structure
                contrast_info = self._analyze_contrast(region, r)
                print(f"Contraste centre/périphérie: {contrast_info['contrast']:.3f}")
                print(f"Centre plus clair: {contrast_info['center_brighter']}")
                
                # Créer une visualisation de l'analyse
                analysis_img = self._create_analysis_visualization(region, r, dominant_color)
                axes[1, i].imshow(analysis_img)
                axes[1, i].set_title(f'L{best_line} d={distance:.2f}')
                axes[1, i].axis('off')
            else:
                axes[0, i].text(0.5, 0.5, 'Région\ninvalide', ha='center', va='center')
                axes[0, i].set_xlim(0, 1)
                axes[1, i].set_ylim(0, 1)
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _extract_circle_region(self, image, x, y, r):
        """
        Extrait la région autour du cercle
        """
        h, w = image.shape[:2]
        margin = 5
        x1 = max(0, x - r - margin)
        x2 = min(w, x + r + margin)
        y1 = max(0, y - r - margin)
        y2 = min(h, y + r + margin)
        
        if x2 > x1 and y2 > y1:
            return image[y1:y2, x1:x2]
        return None
    
    def _get_dominant_color(self, region, radius):
        """
        Extrait la couleur dominante de l'anneau externe
        """
        h, w = region.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Créer masque annulaire
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Zone annulaire ajustée
        ring_mask = ((distances >= radius * 0.5) & (distances <= radius * 0.9))
        
        if np.any(ring_mask):
            ring_pixels = region[ring_mask]
            return np.median(ring_pixels, axis=0)
        else:
            # Fallback: couleur moyenne de toute la région
            return np.mean(region.reshape(-1, 3), axis=0)
    
    def _find_closest_line(self, color):
        """
        Trouve la ligne de métro la plus proche
        """
        min_distance = float('inf')
        best_line = -1
        
        for line, ref_color in self.ligne_colors.items():
            distance = np.linalg.norm(color - np.array(ref_color))
            if distance < min_distance:
                min_distance = distance
                best_line = line
        
        return best_line, min_distance
    
    def _analyze_contrast(self, region, radius):
        """
        Analyse le contraste centre/périphérie
        """
        gray = color.rgb2gray(region)
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        center_mask = distances <= radius * 0.3
        ring_mask = ((distances >= radius * 0.6) & (distances <= radius * 0.9))
        
        if np.any(center_mask) and np.any(ring_mask):
            center_bright = np.mean(gray[center_mask])
            ring_bright = np.mean(gray[ring_mask])
            contrast = center_bright - ring_bright
            return {
                'contrast': contrast,
                'center_brighter': contrast > 0.1,
                'center_brightness': center_bright,
                'ring_brightness': ring_bright
            }
        
        return {'contrast': 0, 'center_brighter': False}
    
    def _create_analysis_visualization(self, region, radius, dominant_color):
        """
        Crée une visualisation de l'analyse
        """
        # Créer une image composite
        h, w = region.shape[:2]
        analysis = np.zeros((h, w*3, 3))
        
        # Image originale
        analysis[:, :w] = region
        
        # Couleur dominante
        analysis[:, w:2*w] = dominant_color
        
        # Masques d'analyse
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        mask_viz = np.zeros((h, w, 3))
        center_mask = distances <= radius * 0.3
        ring_mask = ((distances >= radius * 0.6) & (distances <= radius * 0.9))
        
        mask_viz[center_mask] = [1, 0, 0]  # Rouge pour centre
        mask_viz[ring_mask] = [0, 1, 0]    # Vert pour anneau
        
        analysis[:, 2*w:] = mask_viz
        
        return analysis


# Fonction principale de debug
def debug_metro_detection(image_path):
    """
    Lance le debug complet
    """
    debugger = MetroDebugger()
    circles = debugger.debug_full_pipeline(image_path)
    return circles


# Test rapide de couleurs
def test_color_extraction():
    """
    Test rapide pour voir si l'extraction de couleur fonctionne
    """
    # Créer une image test avec des cercles colorés
    test_img = np.ones((200, 200, 3)) * 0.5  # Fond gris
    
    # Ajouter un cercle ligne 4 (rose)
    center = (100, 100)
    for y in range(200):
        for x in range(200):
            dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
            if 30 <= dist <= 40:  # Anneau
                test_img[y, x] = [0.753, 0.255, 0.569]  # Couleur ligne 4
            elif dist <= 25:  # Centre
                test_img[y, x] = [0.9, 0.9, 0.9]  # Blanc
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_img)
    plt.title("Image test")
    
    # Tester l'extraction
    debugger = MetroDebugger()
    dominant_color = debugger._get_dominant_color(test_img, 35)
    best_line, distance = debugger._find_closest_line(dominant_color)
    
    plt.subplot(1, 2, 2)
    plt.imshow([[dominant_color]])
    plt.title(f"Couleur extraite\nLigne {best_line}, distance: {distance:.3f}")
    
    plt.show()
    
    print(f"Couleur extraite: {dominant_color}")
    print(f"Ligne détectée: {best_line}")
    print(f"Distance: {distance}")


if __name__ == "__main__":
    # Test rapide
    test_color_extraction()
    
    # Pour debugger une vraie image:
    circles = debug_metro_detection("../BD_METRO/IM (9).jpg")
    print("Programme de debug prêt!")
    print("Utilisez debug_metro_detection('path/to/image.jpg') pour analyser une image")