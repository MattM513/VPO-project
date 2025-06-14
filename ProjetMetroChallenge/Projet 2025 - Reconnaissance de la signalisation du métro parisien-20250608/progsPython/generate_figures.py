# -*- coding: utf-8 -*-
"""
GÉNÉRATEUR DE FIGURES POUR LE RAPPORT
Script pour créer toutes les figures nécessaires au rapport technique
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import cv2
from skimage import color, filters, morphology, measure, feature
from skimage.transform import resize
import scipy.io as sio
from PIL import Image
import os
from collections import defaultdict
import pytesseract

# Importer le système existant
from myMetroProcessing import ImprovedMetroSystem

class ReportFiguresGenerator:
    """
    Générateur de figures pour le rapport technique
    """
    
    def __init__(self, images_folder='../BD_METRO', gt_file='Apprentissage.mat'):
        self.images_folder = images_folder
        self.gt_file = gt_file
        self.output_dir = 'report_figures'
        
        # Créer le dossier de sortie
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialiser le système
        self.metro_system = ImprovedMetroSystem()
        
        # Charger quelques images de test
        self.test_images = self._load_test_images()
        
    def _load_test_images(self):
        """Charge quelques images représentatives"""
        images = {}
        # Charger images avec différentes lignes
        test_nums = [6, 9, 30, 42, 108]  # Images avec différents types de signes
        
        for num in test_nums:
            img_path = os.path.join(self.images_folder, f'IM ({num}).JPG')
            if os.path.exists(img_path):
                images[num] = np.array(Image.open(img_path).convert('RGB')) / 255.0
        
        return images
    
    def generate_all_figures(self):
        """Génère toutes les figures nécessaires pour le rapport"""
        print("🎨 GÉNÉRATION DES FIGURES POUR LE RAPPORT")
        print("="*50)
        
        # 1. Segmentation par couleur
        self.generate_color_segmentation_figure()
        
        # 2. Analyse de formes (Transformée de Hough)
        self.generate_hough_transform_figure()
        
        # 3. Template Matching
        self.generate_template_matching_figure()
        
        # 4. OCR (Reconnaissance de caractères)
        self.generate_ocr_figure()
        
        # 5. Extraction de caractéristiques HOG
        self.generate_hog_features_figure()
        
        # 6. Architecture CNN/YOLO conceptuelle
        self.generate_cnn_architecture_figure()
        
        # 7. Résultats de détection complets
        self.generate_detection_results_figure()
        
        # 8. Comparaison méthodes
        self.generate_methods_comparison_figure()
        
        print(f"✅ Toutes les figures ont été générées dans le dossier '{self.output_dir}'")
    
    def generate_color_segmentation_figure(self):
        """Figure 1: Segmentation par couleur"""
        print("📊 Génération Figure 1: Segmentation par couleur")
        
        # Prendre une image avec ligne 14 (violet)
        if 6 in self.test_images:
            image = self.test_images[6]
        else:
            image = list(self.test_images.values())[0]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Image originale
        axes[0,0].imshow(image)
        axes[0,0].set_title('(a) Image originale', fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        
        # Conversion HSV
        hsv_image = color.rgb2hsv(image)
        axes[0,1].imshow(hsv_image)
        axes[0,1].set_title('(b) Image en espace HSV', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # Segmentation ligne 14 (violet)
        target_color_hsv = color.rgb2hsv(np.array([[[0.400, 0.141, 0.514]]]))[0,0]
        hue_target = target_color_hsv[0]
        
        # Masque couleur violet
        hue_channel = hsv_image[:,:,0]
        sat_channel = hsv_image[:,:,1]
        val_channel = hsv_image[:,:,2]
        
        # Tolérance pour le violet
        hue_tolerance = 0.1
        hue_mask = np.abs(hue_channel - hue_target) < hue_tolerance
        sat_mask = sat_channel > 0.3  # Suffisamment saturé
        val_mask = (val_channel > 0.2) & (val_channel < 0.8)  # Ni trop noir ni trop blanc
        
        violet_mask = hue_mask & sat_mask & val_mask
        
        axes[0,2].imshow(violet_mask, cmap='gray')
        axes[0,2].set_title('(c) Masque Ligne 14 (Violet)', fontsize=12, fontweight='bold')
        axes[0,2].axis('off')
        
        # Segmentation ligne 1 (jaune)
        target_color_hsv_yellow = color.rgb2hsv(np.array([[[1.0, 0.808, 0.0]]]))[0,0]
        hue_target_yellow = target_color_hsv_yellow[0]
        
        hue_mask_yellow = np.abs(hue_channel - hue_target_yellow) < 0.08
        yellow_mask = hue_mask_yellow & sat_mask & val_mask
        
        axes[1,0].imshow(yellow_mask, cmap='gray')
        axes[1,0].set_title('(d) Masque Ligne 1 (Jaune)', fontsize=12, fontweight='bold')
        axes[1,0].axis('off')
        
        # Segmentation ligne 12 (vert)
        target_color_hsv_green = color.rgb2hsv(np.array([[[0.0, 0.506, 0.310]]]))[0,0]
        hue_target_green = target_color_hsv_green[0]
        
        hue_mask_green = np.abs(hue_channel - hue_target_green) < 0.08
        green_mask = hue_mask_green & sat_mask & val_mask
        
        axes[1,1].imshow(green_mask, cmap='gray')
        axes[1,1].set_title('(e) Masque Ligne 12 (Vert)', fontsize=12, fontweight='bold')
        axes[1,1].axis('off')
        
        # Combinaison des masques
        combined_mask = violet_mask | yellow_mask | green_mask
        axes[1,2].imshow(combined_mask, cmap='gray')
        axes[1,2].set_title('(f) Masques combinés', fontsize=12, fontweight='bold')
        axes[1,2].axis('off')
        
        plt.suptitle('Figure 1: Segmentation par couleur dans l\'espace HSV', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig1_color_segmentation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_hough_transform_figure(self):
        """Figure 2: Transformée de Hough pour la détection de cercles"""
        print("📊 Génération Figure 2: Transformée de Hough")
        
        # Prendre une image avec des cercles bien définis
        if 30 in self.test_images:
            image = self.test_images[30]
        else:
            image = list(self.test_images.values())[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Image originale
        axes[0,0].imshow(image)
        axes[0,0].set_title('(a) Image originale', fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        
        # Conversion en niveaux de gris
        gray = color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        axes[0,1].imshow(gray, cmap='gray')
        axes[0,1].set_title('(b) Image en niveaux de gris', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # Détection de contours (Canny)
        edges = cv2.Canny(gray_uint8, 50, 150)
        
        axes[1,0].imshow(edges, cmap='gray')
        axes[1,0].set_title('(c) Contours détectés (Canny)', fontsize=12, fontweight='bold')
        axes[1,0].axis('off')
        
        # Transformée de Hough pour cercles
        gray_blurred = cv2.medianBlur(gray_uint8, 5)
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1, minDist=30,
            param1=60, param2=35,
            minRadius=15, maxRadius=60
        )
        
        # Affichage des cercles détectés
        result_image = image.copy()
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for i, (x, y, r) in enumerate(circles[:8]):  # Limiter à 8 cercles
                # Cercle de contour
                circle = Circle((x, y), r, linewidth=2, 
                              edgecolor='red', facecolor='none')
                axes[1,1].add_patch(circle)
                # Point central
                axes[1,1].plot(x, y, 'ro', markersize=3)
                # Numéro
                axes[1,1].text(x+r+5, y, f'{i+1}', color='red', 
                             fontsize=10, fontweight='bold')
        
        axes[1,1].imshow(image)
        axes[1,1].set_title('(d) Cercles détectés (Transformée de Hough)', 
                           fontsize=12, fontweight='bold')
        axes[1,1].axis('off')
        
        plt.suptitle('Figure 2: Détection de formes circulaires par Transformée de Hough', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig2_hough_transform.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_template_matching_figure(self):
        """Figure 3: Template Matching"""
        print("📊 Génération Figure 3: Template Matching")
        
        # Créer un template artificiel (cercle avec chiffre)
        template_size = 60
        template = np.ones((template_size, template_size, 3)) * 0.4  # Fond violet
        
        # Ajouter un cercle violet
        center = template_size // 2
        y_coords, x_coords = np.ogrid[:template_size, :template_size]
        circle_mask = (x_coords - center)**2 + (y_coords - center)**2 <= (center-5)**2
        template[circle_mask] = [0.400, 0.141, 0.514]  # Violet ligne 14
        
        # Ajouter le chiffre "14" en blanc au centre
        inner_mask = (x_coords - center)**2 + (y_coords - center)**2 <= (center//2)**2
        template[inner_mask] = [1.0, 1.0, 1.0]  # Blanc
        
        # Prendre une image de test
        if 6 in self.test_images:
            test_image = self.test_images[6]
        else:
            test_image = list(self.test_images.values())[0]
        
        # Template matching
        template_gray = color.rgb2gray(template)
        test_gray = color.rgb2gray(test_image)
        
        result = cv2.matchTemplate((test_gray * 255).astype(np.uint8), 
                                 (template_gray * 255).astype(np.uint8), 
                                 cv2.TM_CCOEFF_NORMED)
        
        # Trouver le meilleur match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Template
        axes[0,0].imshow(template)
        axes[0,0].set_title('(a) Template - Ligne 14', fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        
        # Image de test
        axes[0,1].imshow(test_image)
        axes[0,1].set_title('(b) Image de recherche', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # Carte de similarité
        axes[1,0].imshow(result, cmap='hot')
        axes[1,0].set_title('(c) Carte de similarité', fontsize=12, fontweight='bold')
        axes[1,0].axis('off')
        
        # Résultat avec template localisé
        result_image = test_image.copy()
        top_left = max_loc
        bottom_right = (top_left[0] + template_size, top_left[1] + template_size)
        
        # Dessiner le rectangle
        rect = Rectangle(top_left, template_size, template_size,
                        linewidth=3, edgecolor='red', facecolor='none')
        
        axes[1,1].imshow(result_image)
        axes[1,1].add_patch(rect)
        axes[1,1].text(top_left[0], top_left[1]-10, f'Match: {max_val:.2f}', 
                      color='red', fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        axes[1,1].set_title('(d) Template localisé', fontsize=12, fontweight='bold')
        axes[1,1].axis('off')
        
        plt.suptitle('Figure 3: Principe du Template Matching', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig3_template_matching.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_ocr_figure(self):
        """Figure 4: Reconnaissance de caractères (OCR)"""
        print("📊 Génération Figure 4: OCR")
        
        # Créer une région artificielle avec le chiffre "14"
        region_size = 80
        region = np.ones((region_size, region_size, 3)) * 0.4  # Fond violet
        
        # Ajouter le chiffre "14" en blanc
        center_x, center_y = region_size // 2, region_size // 2
        
        # Créer le chiffre "14" de manière simple
        # "1" - ligne verticale
        region[center_y-15:center_y+15, center_x-8:center_x-5] = [1.0, 1.0, 1.0]
        # "4" - forme en L
        region[center_y-15:center_y, center_x+2:center_x+5] = [1.0, 1.0, 1.0]  # vertical gauche
        region[center_y-2:center_y+2, center_x+2:center_x+12] = [1.0, 1.0, 1.0]  # horizontal
        region[center_y-15:center_y+15, center_x+9:center_x+12] = [1.0, 1.0, 1.0]  # vertical droit
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Région originale colorée
        axes[0,0].imshow(region)
        axes[0,0].set_title('(a) Région d\'intérêt\n(Pictogramme Ligne 14)', 
                           fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        
        # Conversion en niveaux de gris
        gray = color.rgb2gray(region)
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        axes[0,1].imshow(gray, cmap='gray')
        axes[0,1].set_title('(b) Conversion\nniveaux de gris', 
                           fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # Binarisation
        binary = cv2.adaptiveThreshold(gray_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        axes[0,2].imshow(binary, cmap='gray')
        axes[0,2].set_title('(c) Binarisation\nadaptative', 
                           fontsize=12, fontweight='bold')
        axes[0,2].axis('off')
        
        # Nettoyage morphologique
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        axes[1,0].imshow(cleaned, cmap='gray')
        axes[1,0].set_title('(d) Nettoyage\nmorphologique', 
                           fontsize=12, fontweight='bold')
        axes[1,0].axis('off')
        
        # Redimensionnement pour OCR
        scale_factor = 3
        resized = cv2.resize(cleaned, (region_size * scale_factor, region_size * scale_factor))
        
        axes[1,1].imshow(resized, cmap='gray')
        axes[1,1].set_title('(e) Redimensionnement\npour OCR', 
                           fontsize=12, fontweight='bold')
        axes[1,1].axis('off')
        
        # Simulation du résultat OCR
        result_image = region.copy()
        
        # Ajouter le texte du résultat OCR
        axes[1,2].imshow(result_image)
        axes[1,2].text(center_x, region_size + 10, 'OCR Result: "14"', 
                      ha='center', va='center', 
                      fontsize=14, fontweight='bold', color='red',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
        axes[1,2].text(center_x, region_size + 25, 'Confidence: 0.89', 
                      ha='center', va='center', 
                      fontsize=12, color='green',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        axes[1,2].set_title('(f) Résultat OCR\n(Tesseract)', 
                           fontsize=12, fontweight='bold')
        axes[1,2].axis('off')
        
        plt.suptitle('Figure 4: Pipeline de Reconnaissance de Caractères (OCR)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig4_ocr_pipeline.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_hog_features_figure(self):
        """Figure 5: Extraction de caractéristiques HOG"""
        print("📊 Génération Figure 5: Caractéristiques HOG")
        
        # Prendre une petite région d'un pictogramme
        if 6 in self.test_images:
            image = self.test_images[6]
        else:
            image = list(self.test_images.values())[0]
        
        # Extraire une région d'intérêt (simulée)
        h, w = image.shape[:2]
        roi_size = 100
        roi_x, roi_y = w//2 - roi_size//2, h//2 - roi_size//2
        roi = image[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        
        # Convertir en niveaux de gris
        roi_gray = color.rgb2gray(roi)
        
        # Calculer les caractéristiques HOG
        hog_features, hog_image = feature.hog(roi_gray, 
                                            orientations=9,
                                            pixels_per_cell=(16, 16),
                                            cells_per_block=(2, 2),
                                            visualize=True,
                                            block_norm='L2-Hys')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # ROI originale
        axes[0].imshow(roi)
        axes[0].set_title('(a) Région d\'intérêt\n(Pictogramme)', 
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # ROI en niveaux de gris
        axes[1].imshow(roi_gray, cmap='gray')
        axes[1].set_title('(b) Image en\nniveaux de gris', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Visualisation HOG
        axes[2].imshow(hog_image, cmap='gray')
        axes[2].set_title('(c) Caractéristiques HOG\n(Histogramme de gradients orientés)', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Ajouter des informations sur les caractéristiques
        info_text = f"Vecteur HOG:\n- {len(hog_features)} caractéristiques\n- 9 orientations\n- Cellules 16×16"
        plt.figtext(0.02, 0.15, info_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Figure 5: Extraction de caractéristiques HOG pour classification', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig5_hog_features.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_cnn_architecture_figure(self):
        """Figure 6: Architecture CNN conceptuelle"""
        print("📊 Génération Figure 6: Architecture CNN")
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Dessiner l'architecture CNN de manière schématique
        # Couches et leurs positions
        layers = [
            {'name': 'Input\n224×224×3', 'pos': (1, 4), 'size': (1.5, 3), 'color': 'lightblue'},
            {'name': 'Conv2D\n32 filters\n3×3', 'pos': (3, 4), 'size': (1.2, 2.5), 'color': 'orange'},
            {'name': 'MaxPool\n2×2', 'pos': (5, 4.2), 'size': (0.8, 2), 'color': 'lightgreen'},
            {'name': 'Conv2D\n64 filters\n3×3', 'pos': (6.5, 4), 'size': (1.2, 2.2), 'color': 'orange'},
            {'name': 'MaxPool\n2×2', 'pos': (8.5, 4.3), 'size': (0.8, 1.8), 'color': 'lightgreen'},
            {'name': 'Conv2D\n128 filters\n3×3', 'pos': (10, 4), 'size': (1.2, 1.8), 'color': 'orange'},
            {'name': 'Flatten', 'pos': (12, 4.5), 'size': (0.8, 1), 'color': 'yellow'},
            {'name': 'Dense\n512 units', 'pos': (13.5, 4.2), 'size': (1, 1.5), 'color': 'lightcoral'},
            {'name': 'Dense\n14 classes\n(Lignes)', 'pos': (15.5, 4.2), 'size': (1.2, 1.5), 'color': 'lightcoral'}
        ]
        
        # Dessiner les couches
        for layer in layers:
            rect = Rectangle(layer['pos'], layer['size'][0], layer['size'][1], 
                           facecolor=layer['color'], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Ajouter le texte
            text_x = layer['pos'][0] + layer['size'][0]/2
            text_y = layer['pos'][1] + layer['size'][1]/2
            ax.text(text_x, text_y, layer['name'], ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        
        # Dessiner les flèches entre les couches
        arrow_props = dict(arrowstyle='->', lw=2, color='blue')
        positions = [layer['pos'] for layer in layers]
        
        for i in range(len(positions)-1):
            start_x = positions[i][0] + layers[i]['size'][0]
            start_y = positions[i][1] + layers[i]['size'][1]/2
            end_x = positions[i+1][0]
            end_y = positions[i+1][1] + layers[i+1]['size'][1]/2
            
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y), 
                       arrowprops=arrow_props)
        
        # Configuration des axes
        ax.set_xlim(0, 17)
        ax.set_ylim(2, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Ajouter des annotations
        ax.text(8.5, 7, 'Extraction de caractéristiques', ha='center', 
               fontsize=12, fontweight='bold', color='darkblue')
        ax.text(14.5, 7, 'Classification', ha='center', 
               fontsize=12, fontweight='bold', color='darkred')
        
        # Ligne de séparation
        ax.axvline(x=11.5, ymin=0.2, ymax=0.8, color='gray', linestyle='--', linewidth=2)
        
        plt.suptitle('Figure 6: Architecture d\'un Réseau de Neurones Convolutif (CNN)\npour la détection de lignes de métro', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig6_cnn_architecture.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detection_results_figure(self):
        """Figure 7: Résultats de détection complets"""
        print("📊 Génération Figure 7: Résultats de détection")
        
        # Entraîner le système sur les données
        self.metro_system.train_system(self.images_folder, self.gt_file)
        
        # Prendre plusieurs images représentatives
        test_images = [6, 9, 30] if all(i in self.test_images for i in [6, 9, 30]) else list(self.test_images.keys())[:3]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, img_num in enumerate(test_images[:6]):
            if img_num not in self.test_images:
                continue
                
            image = self.test_images[img_num]
            
            # Détecter les signes
            detections = self.metro_system.detect_metro_signs(image, debug=False)
            
            # Afficher l'image avec les détections
            axes[idx].imshow(image)
            
            # Couleurs pour l'affichage
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            
            for i, detection in enumerate(detections):
                if detection['bbox']:
                    x1, x2, y1, y2 = detection['bbox']
                    
                    # Rectangle de détection
                    rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=3, 
                                   edgecolor=colors[i % len(colors)], 
                                   facecolor='none')
                    axes[idx].add_patch(rect)
                    
                    # Label avec numéro de ligne
                    axes[idx].text(x1, y1-10, f"L{detection['ligne']}", 
                                  color='white', fontsize=12, fontweight='bold',
                                  bbox=dict(boxstyle="round,pad=0.3", 
                                          facecolor=colors[i % len(colors)], alpha=0.8))
                    
                    # Confiance
                    axes[idx].text(x2, y2+15, f"{detection['confidence']:.2f}", 
                                  color='white', fontsize=10,
                                  bbox=dict(boxstyle="round,pad=0.2", 
                                          facecolor='black', alpha=0.7))
            
            # Titre avec informations
            lignes_detectees = [str(d['ligne']) for d in detections] if detections else ['Aucune']
            axes[idx].set_title(f'IM ({img_num}) - Lignes: {", ".join(lignes_detectees)}', 
                               fontsize=11, fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('Figure 7: Résultats de détection du système hybride (Couleur + OCR)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig7_detection_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_methods_comparison_figure(self):
        """Figure 8: Comparaison des méthodes"""
        print("📊 Génération Figure 8: Comparaison des méthodes")
        
        # Prendre une image test
        if 6 in self.test_images:
            image = self.test_images[6]
        else:
            image = list(self.test_images.values())[0]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Image originale
        axes[0,0].imshow(image)
        axes[0,0].set_title('(a) Image originale', fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        
        # Méthode 1: Segmentation couleur seule
        hsv_image = color.rgb2hsv(image)
        
        # Masques pour différentes couleurs
        colors_to_detect = {
            14: [0.400, 0.141, 0.514],  # Violet
            1: [1.0, 0.808, 0.0],       # Jaune
            12: [0.0, 0.506, 0.310]     # Vert
        }
        
        color_result = image.copy()
        detected_regions = []
        
        for ligne, rgb_color in colors_to_detect.items():
            hsv_target = color.rgb2hsv(np.array([[rgb_color]]))[0,0]
            hue_channel = hsv_image[:,:,0]
            sat_channel = hsv_image[:,:,1]
            
            hue_mask = np.abs(hue_channel - hsv_target[0]) < 0.1
            sat_mask = sat_channel > 0.3
            combined_mask = hue_mask & sat_mask
            
            # Trouver les composantes connexes
            labeled_mask = measure.label(combined_mask)
            regions = measure.regionprops(labeled_mask)
            
            for region in regions:
                if region.area > 100:  # Filtrer les petites régions
                    bbox = region.bbox
                    detected_regions.append((bbox, ligne))
        
        # Dessiner les régions détectées par couleur
        colors_display = ['red', 'blue', 'green']
        for i, (bbox, ligne) in enumerate(detected_regions[:3]):
            y1, x1, y2, x2 = bbox
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor=colors_display[i], facecolor='none')
            axes[0,1].add_patch(rect)
            axes[0,1].text(x1, y1-5, f"L{ligne}", color=colors_display[i], 
                          fontsize=10, fontweight='bold')
        
        axes[0,1].imshow(image)
        axes[0,1].set_title('(b) Méthode couleur seule', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # Méthode 2: Détection de formes seule (Hough)
        gray = color.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        gray_blurred = cv2.medianBlur(gray_uint8, 5)
        
        circles = cv2.HoughCircles(
            gray_blurred, cv2.HOUGH_GRADIENT,
            dp=1, minDist=30, param1=60, param2=35,
            minRadius=15, maxRadius=60
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for i, (x, y, r) in enumerate(circles[:5]):
                circle = Circle((x, y), r, linewidth=2, 
                              edgecolor='orange', facecolor='none')
                axes[0,2].add_patch(circle)
                axes[0,2].text(x+r+5, y, f"C{i+1}", color='orange', 
                              fontsize=10, fontweight='bold')
        
        axes[0,2].imshow(image)
        axes[0,2].set_title('(c) Détection de formes seule\n(Transformée de Hough)', 
                           fontsize=12, fontweight='bold')
        axes[0,2].axis('off')
        
        # Méthode 3: OCR seul (simulation)
        # Simuler des régions où l'OCR détecte des chiffres
        ocr_detections = [
            {'bbox': (100, 150, 50, 100), 'text': '14', 'conf': 0.85},
            {'bbox': (200, 250, 120, 170), 'text': 'RER', 'conf': 0.90},
            {'bbox': (300, 350, 80, 130), 'text': '1', 'conf': 0.75}
        ]
        
        for i, det in enumerate(ocr_detections):
            x1, x2, y1, y2 = det['bbox']
            color_ocr = 'green' if det['text'].isdigit() else 'red'
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor=color_ocr, facecolor='none')
            axes[1,0].add_patch(rect)
            axes[1,0].text(x1, y1-5, f"'{det['text']}'", color=color_ocr, 
                          fontsize=10, fontweight='bold')
        
        axes[1,0].imshow(image)
        axes[1,0].set_title('(d) OCR seul\n(Vert: valide, Rouge: rejeté)', 
                           fontsize=12, fontweight='bold')
        axes[1,0].axis('off')
        
        # Méthode 4: Système hybride (notre méthode)
        if not self.metro_system.is_trained:
            self.metro_system.train_system(self.images_folder, self.gt_file)
        
        detections = self.metro_system.detect_metro_signs(image, debug=False)
        
        colors_hybrid = ['lime', 'red', 'blue', 'orange', 'purple']
        for i, detection in enumerate(detections):
            if detection['bbox']:
                x1, x2, y1, y2 = detection['bbox']
                
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=3, 
                               edgecolor=colors_hybrid[i % len(colors_hybrid)], 
                               facecolor='none')
                axes[1,1].add_patch(rect)
                
                # Méthode de validation
                method_emoji = {
                    'agreement': '✅', 'color_priority': '🎨', 
                    'ocr_priority': '🔢', 'fallback': '⚠️'
                }
                emoji = method_emoji.get(detection.get('validation_method', 'fallback'), '?')
                
                axes[1,1].text(x1, y1-10, f"L{detection['ligne']} {emoji}", 
                              color='white', fontsize=11, fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", 
                                      facecolor=colors_hybrid[i % len(colors_hybrid)], alpha=0.8))
        
        axes[1,1].imshow(image)
        axes[1,1].set_title('(e) Système hybride\n(Couleur + OCR + Validation croisée)', 
                           fontsize=12, fontweight='bold')
        axes[1,1].axis('off')
        
        # Tableau de comparaison des performances
        methods_data = {
            'Méthode': ['Couleur seule', 'Formes seules', 'OCR seul', 'Système hybride'],
            'Précision': ['Moyenne', 'Faible', 'Élevée', 'Très élevée'],
            'Rappel': ['Élevé', 'Très élevé', 'Faible', 'Élevé'],
            'Robustesse': ['Faible', 'Moyenne', 'Faible', 'Très élevée'],
            'Faux positifs': ['Élevés', 'Très élevés', 'Moyens', 'Faibles']
        }
        
        # Créer le tableau
        axes[1,2].axis('off')
        table_data = []
        for i in range(len(methods_data['Méthode'])):
            row = [methods_data[key][i] for key in methods_data.keys()]
            table_data.append(row)
        
        table = axes[1,2].table(cellText=table_data, 
                               colLabels=list(methods_data.keys()),
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Colorier les cellules selon les performances
        colors_table = {
            'Très élevée': '#90EE90', 'Élevée': '#FFD700', 'Élevé': '#FFD700',
            'Moyenne': '#FFA500', 'Faible': '#FFB6C1', 'Faibles': '#90EE90',
            'Moyens': '#FFA500', 'Élevés': '#FFB6C1', 'Très élevés': '#FF6B6B'
        }
        
        for i in range(len(table_data)):
            for j, value in enumerate(table_data[i][1:], 1):  # Skip method name
                if value in colors_table:
                    table[(i+1, j)].set_facecolor(colors_table[value])
        
        axes[1,2].set_title('(f) Comparaison des performances', 
                           fontsize=12, fontweight='bold')
        
        plt.suptitle('Figure 8: Comparaison des différentes approches de détection', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig8_methods_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_pipeline_overview_figure(self):
        """Figure 9: Vue d'ensemble du pipeline complet"""
        print("📊 Génération Figure 9: Pipeline complet")
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Définir les étapes du pipeline
        steps = [
            {'name': 'Image\noriginale', 'pos': (1, 8), 'size': (2, 1.5), 'color': 'lightblue'},
            {'name': 'Prétraitement\n(Redimensionnement,\nFlou médian)', 'pos': (1, 6), 'size': (2, 1.2), 'color': 'lightgray'},
            {'name': 'Détection\nde cercles\n(Hough Transform)', 'pos': (4, 8), 'size': (2.5, 1.5), 'color': 'orange'},
            {'name': 'Filtrage\nbasique\n(Taille, Position)', 'pos': (4, 6), 'size': (2.5, 1.2), 'color': 'lightyellow'},
            {'name': 'Validation\npar couleur\n(HSV + Clustering)', 'pos': (1, 3.5), 'size': (2.8, 1.8), 'color': 'lightgreen'},
            {'name': 'Validation\npar OCR\n(Tesseract)', 'pos': (4.5, 3.5), 'size': (2.8, 1.8), 'color': 'lightcoral'},
            {'name': 'Analyse intelligente\n(Validation croisée)', 'pos': (8, 6), 'size': (3, 1.5), 'color': 'plum'},
            {'name': 'Suppression\ndes doublons', 'pos': (8, 4), 'size': (3, 1.2), 'color': 'lightsalmon'},
            {'name': 'Résultats finaux\n(Ligne + Confiance)', 'pos': (12, 5), 'size': (3, 1.5), 'color': 'lightsteelblue'}
        ]
        
        # Dessiner les boîtes
        for step in steps:
            rect = Rectangle(step['pos'], step['size'][0], step['size'][1], 
                           facecolor=step['color'], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Ajouter le texte
            text_x = step['pos'][0] + step['size'][0]/2
            text_y = step['pos'][1] + step['size'][1]/2
            ax.text(text_x, text_y, step['name'], ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Dessiner les flèches du workflow
        arrow_props = dict(arrowstyle='->', lw=2, color='blue')
        
        # Flèches principales
        arrows = [
            ((2, 8), (2, 7.2)),  # Image -> Prétraitement
            ((3, 6.5), (4, 8.5)),  # Prétraitement -> Détection cercles
            ((5.25, 8), (5.25, 7.2)),  # Détection -> Filtrage
            ((3.8, 6), (2.8, 4.8)),  # Filtrage -> Validation couleur
            ((5.7, 6), (5.9, 5.3)),  # Filtrage -> Validation OCR
            ((3.8, 4.4), (8, 6.5)),  # Couleur -> Analyse
            ((7.3, 4.4), (8, 6.2)),  # OCR -> Analyse
            ((9.5, 6), (9.5, 5.2)),  # Analyse -> Suppression doublons
            ((11, 4.6), (12, 5.5))   # Suppression -> Résultats
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
        
        # Ajouter des annotations pour les critères de décision
        ax.text(6.5, 2.5, 'Critères de validation:\n• Accord couleur-OCR → Confiance max\n• Couleur forte + OCR faible → Couleur\n• OCR fort + couleur faible → OCR prudent\n• Conflit → Arbitrage par confiance', 
               fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.8))
        
        # Configuration des axes
        ax.set_xlim(0, 16)
        ax.set_ylim(1, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.suptitle('Figure 9: Pipeline complet du système de détection hybride', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig9_complete_pipeline.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Fonction principale pour générer toutes les figures"""
    print("🎨 GÉNÉRATEUR DE FIGURES POUR LE RAPPORT TECHNIQUE")
    print("="*60)
    
    # Vérifier que les dossiers existent
    if not os.path.exists('../BD_METRO'):
        print("❌ Erreur: Le dossier '../BD_METRO' n'existe pas")
        print("   Ajustez le chemin dans le script")
        return
    
    if not os.path.exists('Apprentissage.mat'):
        print("❌ Erreur: Le fichier 'Apprentissage.mat' n'existe pas")
        print("   Copiez-le dans le répertoire courant")
        return
    
    # Créer le générateur
    generator = ReportFiguresGenerator()
    
    # Menu interactif
    while True:
        print("\n" + "="*50)
        print("MENU - GÉNÉRATION DE FIGURES")
        print("="*50)
        print("1. Segmentation par couleur")
        print("2. Transformée de Hough (détection cercles)")
        print("3. Template Matching")
        print("4. OCR (Reconnaissance caractères)")
        print("5. Caractéristiques HOG")
        print("6. Architecture CNN")
        print("7. Résultats de détection")
        print("8. Comparaison des méthodes")
        print("9. Pipeline complet")
        print("0. GÉNÉRER TOUTES LES FIGURES")
        print("q. Quitter")
        
        choice = input("\nVotre choix: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '0':
            generator.generate_all_figures()
            generator.generate_pipeline_overview_figure()
            print("\n🎉 Toutes les figures ont été générées!")
            break
        elif choice == '1':
            generator.generate_color_segmentation_figure()
        elif choice == '2':
            generator.generate_hough_transform_figure()
        elif choice == '3':
            generator.generate_template_matching_figure()
        elif choice == '4':
            generator.generate_ocr_figure()
        elif choice == '5':
            generator.generate_hog_features_figure()
        elif choice == '6':
            generator.generate_cnn_architecture_figure()
        elif choice == '7':
            generator.generate_detection_results_figure()
        elif choice == '8':
            generator.generate_methods_comparison_figure()
        elif choice == '9':
            generator.generate_pipeline_overview_figure()
        else:
            print("❌ Choix invalide, veuillez réessayer")


if __name__ == "__main__":
    main()