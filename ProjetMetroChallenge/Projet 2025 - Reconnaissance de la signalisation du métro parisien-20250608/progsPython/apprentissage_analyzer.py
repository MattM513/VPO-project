import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.io as sio

def rgb_to_hsv_simple(rgb):
    """Conversion RGB vers HSV simplifiée"""
    r, g, b = rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Valeur
    v = max_val
    
    # Saturation
    s = 0 if max_val == 0 else diff / max_val
    
    # Teinte (approximation)
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    return [h/360, s, v]

class ApprentissageAnalyzer:
    """
    Analyseur des données d'apprentissage pour optimiser la détection
    """
    
    def __init__(self, apprentissage_file='Apprentissage.mat', images_dir='../BD_METRO'):
        self.apprentissage_file = apprentissage_file
        self.images_dir = images_dir
        self.data = None
        self.color_stats = {}
        self.size_stats = {}
        
    def load_data(self):
        """Charge les données d'apprentissage avec le BON FORMAT"""
        try:
            if self.apprentissage_file.endswith('.mat'):
                mat_data = sio.loadmat(self.apprentissage_file)
                bd_array = mat_data['BD']
                
                # CORRECTION: Le format réel est [image_num, x1, x2, y1, y2, ligne]
                # Pas [image_num, x1, y1, x2, y2, ligne] comme je pensais
                self.data = pd.DataFrame(bd_array, columns=['image_num', 'x1', 'x2', 'y1', 'y2', 'ligne'])
                print(f"Fichier .mat chargé avec le BON format: x1, x2, y1, y2")
            
            elif self.apprentissage_file.endswith('.xls') or self.apprentissage_file.endswith('.xlsx'):
                self.data = pd.read_excel(self.apprentissage_file)
                print(f"Fichier Excel chargé avec succès")
            
            else:
                self.data = pd.read_csv(self.apprentissage_file)
                print(f"Fichier CSV chargé avec succès")
                
        except FileNotFoundError:
            print(f"Erreur: fichier '{self.apprentissage_file}' non trouvé")
            return False
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return False
        
        print(f"Données chargées: {len(self.data)} signes d'apprentissage")
        print(f"Colonnes: {list(self.data.columns)}")
        print(f"Premières lignes:")
        print(self.data.head())
        return True
    
    def analyze_real_colors(self):
        """
        Analyse les couleurs RÉELLES avec le BON format de coordonnées
        """
        if self.data is None:
            print("Erreur: données non chargées")
            return
        
        real_colors = {i: [] for i in range(1, 15)}
        processed_count = 0
        error_count = 0
        
        print("Analyse des coordonnées avec le bon format...")
        
        for idx, row in self.data.iterrows():
            try:
                if len(row) >= 6:
                    img_num = int(row.iloc[0])
                    
                    # CORRECTION MAJEURE: Le format est x1, x2, y1, y2
                    x1_raw = int(row.iloc[1])  # x1
                    x2_raw = int(row.iloc[2])  # x2  
                    y1_raw = int(row.iloc[3])  # y1
                    y2_raw = int(row.iloc[4])  # y2
                    ligne = int(row.iloc[5])
                    
                    # S'assurer que x1 < x2 et y1 < y2
                    x1 = min(x1_raw, x2_raw)
                    x2 = max(x1_raw, x2_raw)
                    y1 = min(y1_raw, y2_raw)
                    y2 = max(y1_raw, y2_raw)
                    
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Tailles raisonnables pour signes de métro
                    if width < 15 or height < 15 or width > 150 or height > 150 or area > 22500:
                        print(f"Taille aberrante ignorée: {width}x{height} (aire: {area})")
                        error_count += 1
                        continue
                    
                    # Construire le chemin de l'image
                    img_path = os.path.join(self.images_dir, f'IM ({img_num}).JPG')
                    
                    if os.path.exists(img_path):
                        image = np.array(Image.open(img_path).convert('RGB')) / 255.0
                        
                        # Vérifier que les coordonnées sont dans l'image
                        if (x1 >= 0 and y1 >= 0 and 
                            x2 < image.shape[1] and y2 < image.shape[0]):
                            
                            # Extraire la région du signe
                            region = image[y1:y2, x1:x2]
                            
                            if region.size > 0:
                                h, w = region.shape[:2]
                                
                                # Zone centrale (plus représentative)
                                center_h, center_w = h//2, w//2
                                radius = min(h, w) // 4  # Zone plus petite mais plus fiable
                                
                                if radius > 3:
                                    center_region = region[
                                        max(0, center_h-radius):center_h+radius,
                                        max(0, center_w-radius):center_w+radius
                                    ]
                                    
                                    if center_region.size > 0:
                                        # Calculer couleur moyenne et médiane
                                        center_mean = np.mean(center_region.reshape(-1, 3), axis=0)
                                        center_median = np.median(center_region.reshape(-1, 3), axis=0)
                                        
                                        # Analyser la saturation pour choisir la meilleure
                                        center_hsv = rgb_to_hsv_simple(center_mean)
                                        median_hsv = rgb_to_hsv_simple(center_median)
                                        
                                        # Choisir la couleur la plus saturée
                                        if center_hsv[1] > median_hsv[1]:
                                            color_to_use = center_mean
                                            hsv_final = center_hsv
                                        else:
                                            color_to_use = center_median
                                            hsv_final = median_hsv
                                        
                                        # Validation: couleur suffisamment vive
                                        if hsv_final[1] > 0.15 and hsv_final[2] > 0.2:
                                            real_colors[ligne].append(color_to_use)
                                            processed_count += 1
                                            
                                            # Debug pour vérifier
                                            if processed_count <= 15:
                                                hex_color = '#{:02x}{:02x}{:02x}'.format(
                                                    int(color_to_use[0]*255), 
                                                    int(color_to_use[1]*255), 
                                                    int(color_to_use[2]*255)
                                                )
                                                print(f"✅ Ligne {ligne}: {hex_color} "
                                                      f"RGB({color_to_use[0]:.3f}, {color_to_use[1]:.3f}, {color_to_use[2]:.3f}) "
                                                      f"S:{hsv_final[1]:.2f}")
                                        else:
                                            print(f"❌ Couleur trop terne ligne {ligne}: S={hsv_final[1]:.2f}, V={hsv_final[2]:.2f}")
                                            error_count += 1
                        else:
                            print(f"Coordonnées hors image: ({x1},{y1})-({x2},{y2}) pour image {image.shape}")
                            error_count += 1
                    else:
                        print(f"Image non trouvée: {img_path}")
                        error_count += 1
                else:
                    error_count += 1
                    continue
                    
            except Exception as e:
                print(f"Erreur traitement signe {idx}: {e}")
                error_count += 1
                continue
        
        print(f"\n📊 Résultat: {processed_count} signes valides analysés, {error_count} erreurs")
        
        # Calculer statistiques par ligne
        self.color_stats = {}
        for ligne, colors in real_colors.items():
            if colors:
                colors_array = np.array(colors)
                self.color_stats[ligne] = {
                    'mean': np.mean(colors_array, axis=0),
                    'std': np.std(colors_array, axis=0),
                    'median': np.median(colors_array, axis=0),
                    'count': len(colors),
                    'all_colors': colors_array
                }
        
        return self.color_stats

    def analyze_sizes_and_shapes(self):
        """
        Analyse des tailles avec le BON format de coordonnées
        """
        if self.data is None:
            return
        
        widths = []
        heights = []
        areas = []
        aspect_ratios = []
        
        for _, row in self.data.iterrows():
            try:
                if len(row) >= 5:
                    # CORRECTION: format x1, x2, y1, y2
                    x1_raw = int(row.iloc[1])
                    x2_raw = int(row.iloc[2])
                    y1_raw = int(row.iloc[3])
                    y2_raw = int(row.iloc[4])
                    
                    # S'assurer que x1 < x2 et y1 < y2
                    x1 = min(x1_raw, x2_raw)
                    x2 = max(x1_raw, x2_raw)
                    y1 = min(y1_raw, y2_raw)
                    y2 = max(y1_raw, y2_raw)
                    
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Filtrer les tailles raisonnables
                    if (width > 0 and height > 0 and 
                        width < 150 and height < 150 and
                        width > 15 and height > 15 and
                        area < 22500):
                        
                        aspect_ratio = min(width/height, height/width)
                        
                        widths.append(width)
                        heights.append(height)
                        areas.append(area)
                        aspect_ratios.append(aspect_ratio)
            except:
                continue
        
        if widths:
            self.size_stats = {
                'width': {
                    'min': min(widths), 'max': max(widths), 
                    'mean': np.mean(widths), 'std': np.std(widths)
                },
                'height': {
                    'min': min(heights), 'max': max(heights), 
                    'mean': np.mean(heights), 'std': np.std(heights)
                },
                'area': {
                    'min': min(areas), 'max': max(areas), 
                    'mean': np.mean(areas), 'std': np.std(areas)
                },
                'aspect_ratio': {
                    'min': min(aspect_ratios), 'max': max(aspect_ratios), 
                    'mean': np.mean(aspect_ratios), 'std': np.std(aspect_ratios)
                }
            }
        
        return self.size_stats
    
    def generate_optimized_colors(self):
        """Génère les couleurs optimisées"""
        if not self.color_stats:
            print("Erreur: analysez d'abord les couleurs")
            return {}
        
        optimized_colors = {}
        for ligne, stats in self.color_stats.items():
            if stats['count'] > 0:
                optimized_colors[ligne] = stats['median'].tolist()
        
        return optimized_colors
    
    def generate_optimized_params(self):
        """Génère les paramètres optimisés"""
        if not self.size_stats:
            return {}
        
        area_mean = self.size_stats['area']['mean']
        area_std = self.size_stats['area']['std']
        
        optimized_params = {
            'min_area': max(400, int(area_mean - 1.5*area_std)),
            'max_area': int(area_mean + 1.5*area_std),
            'min_width': max(20, int(self.size_stats['width']['mean'] - 1.5*self.size_stats['width']['std'])),
            'max_width': int(self.size_stats['width']['mean'] + 1.5*self.size_stats['width']['std']),
            'min_height': max(20, int(self.size_stats['height']['mean'] - 1.5*self.size_stats['height']['std'])),
            'max_height': int(self.size_stats['height']['mean'] + 1.5*self.size_stats['height']['std']),
            'min_aspect_ratio': max(0.6, self.size_stats['aspect_ratio']['mean'] - 0.15),
            'optimal_area': int(area_mean)
        }
        
        return optimized_params
    
    def print_analysis_results(self):
        """Affiche les résultats"""
        print("\n🎯 === RÉSULTATS DE L'ANALYSE ===")
        
        if self.color_stats:
            print("\n🌈 --- COULEURS RÉELLES OPTIMISÉES ---")
            for ligne in sorted(self.color_stats.keys()):
                stats = self.color_stats[ligne]
                if stats['count'] > 0:
                    color = stats['median']
                    hex_color = '#{:02x}{:02x}{:02x}'.format(
                        int(color[0]*255), int(color[1]*255), int(color[2]*255)
                    )
                    print(f"Ligne {ligne:2d}: {hex_color} RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}) "
                          f"[{stats['count']} échantillons]")
        
        if self.size_stats:
            print("\n📏 --- STATISTIQUES DE TAILLE ---")
            print(f"Largeur: {self.size_stats['width']['min']}-{self.size_stats['width']['max']} "
                  f"(moy: {self.size_stats['width']['mean']:.1f})")
            print(f"Hauteur: {self.size_stats['height']['min']}-{self.size_stats['height']['max']} "
                  f"(moy: {self.size_stats['height']['mean']:.1f})")
            print(f"Aire: {self.size_stats['area']['min']}-{self.size_stats['area']['max']} "
                  f"(moy: {self.size_stats['area']['mean']:.1f})")


def analyze_apprentissage_data():
    """Fonction principale d'analyse"""
    possible_files = ['Apprentissage.mat', 'Apprentissage.xls', 'Apprentissage.xlsx', 'Apprentissage.csv']
    
    analyzer = None
    for filename in possible_files:
        if os.path.exists(filename):
            print(f"📂 Tentative avec le fichier: {filename}")
            analyzer = ApprentissageAnalyzer(apprentissage_file=filename)
            if analyzer.load_data():
                break
    
    if analyzer is None or analyzer.data is None:
        print("❌ Aucun fichier d'apprentissage trouvé!")
        return None, None, None
    
    print("\n🔍 Analyse des couleurs réelles...")
    analyzer.analyze_real_colors()
    
    print("\n📐 Analyse des tailles et formes...")
    analyzer.analyze_sizes_and_shapes()
    
    print("\n⚙️ Génération des paramètres optimisés...")
    optimized_colors = analyzer.generate_optimized_colors()
    optimized_params = analyzer.generate_optimized_params()
    
    analyzer.print_analysis_results()
    
    return analyzer, optimized_colors, optimized_params

# Test direct
if __name__ == "__main__":
    print("🚀 === ANALYSE DES DONNÉES D'APPRENTISSAGE ===")
    analyzer, colors, params = analyze_apprentissage_data()
    
    if colors:
        print(f"\n✅ Analyse réussie! {len(colors)} couleurs optimisées générées")
        print(f"📊 Paramètres optimisés: {params}")
    else:
        print("\n❌ Échec de l'analyse")