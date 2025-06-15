# -*- coding: utf-8 -*-
"""
Analyseur de métriques détaillé pour le challenge métro
Extrait et analyse les F1-score, précision, rappel, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from collections import defaultdict
import json

class MetricsAnalyzer:
    """
    Analyseur de métriques pour les résultats de détection
    """
    
    def __init__(self, gt_file, results_file, resize_factor=1):
        self.gt_file = gt_file
        self.results_file = results_file
        self.resize_factor = resize_factor
        
        # Charger les données
        self.gt_data = self._load_ground_truth()
        self.results_data = self._load_results()
        
        # Calculer les métriques
        self.metrics = self._calculate_detailed_metrics()
    
    def _load_ground_truth(self):
        """Charge la vérité terrain"""
        try:
            gt_mat = sio.loadmat(self.gt_file)
            return gt_mat['BD']
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {self.gt_file}: {e}")
            return np.array([])
    
    def _load_results(self):
        """Charge les résultats de détection"""
        try:
            results_mat = sio.loadmat(self.results_file)
            return results_mat['BD']
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {self.results_file}: {e}")
            return np.array([])
    
    def _calculate_iou(self, box1, box2):
        """Calcule l'IoU entre deux boîtes [x1, x2, y1, y2]"""
        # Coordonnées des boîtes
        x1_1, x2_1, y1_1, y2_1 = box1
        x1_2, x2_2, y1_2, y2_2 = box2
        
        # Intersection
        x1_inter = max(x1_1, x1_2)
        x2_inter = min(x2_1, x2_2)
        y1_inter = max(y1_1, y1_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_detailed_metrics(self):
        """Calcule les métriques détaillées"""
        print("📊 CALCUL DES MÉTRIQUES DÉTAILLÉES")
        print("="*50)
        
        if len(self.gt_data) == 0 or len(self.results_data) == 0:
            print("❌ Données insuffisantes pour calculer les métriques")
            return {}
        
        # Organiser par image
        gt_by_image = defaultdict(list)
        results_by_image = defaultdict(list)
        
        for row in self.gt_data:
            img_num = int(row[0])
            bbox = row[1:5] * self.resize_factor
            ligne = int(row[5])
            gt_by_image[img_num].append({'bbox': bbox, 'ligne': ligne})
        
        for row in self.results_data:
            img_num = int(row[0])
            bbox = row[1:5] * self.resize_factor
            ligne = int(row[5])
            results_by_image[img_num].append({'bbox': bbox, 'ligne': ligne})
        
        # Métriques globales
        total_tp = 0  # Vrais positifs
        total_fp = 0  # Faux positifs
        total_fn = 0  # Faux négatifs
        
        # Métriques par ligne
        line_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Seuil IoU pour considérer une détection comme correcte
        iou_threshold = 0.5
        
        # Analyser chaque image
        for img_num in set(list(gt_by_image.keys()) + list(results_by_image.keys())):
            gt_objects = gt_by_image.get(img_num, [])
            detected_objects = results_by_image.get(img_num, [])
            
            # Marquer les objets GT comme détectés ou non
            gt_matched = [False] * len(gt_objects)
            
            # Pour chaque détection, trouver le meilleur match GT
            for detection in detected_objects:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_obj in enumerate(gt_objects):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = self._calculate_iou(detection['bbox'], gt_obj['bbox'])
                    
                    if (iou > best_iou and iou >= iou_threshold and 
                        detection['ligne'] == gt_obj['ligne']):
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    # Vrai positif
                    total_tp += 1
                    line_metrics[detection['ligne']]['tp'] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    # Faux positif
                    total_fp += 1
                    line_metrics[detection['ligne']]['fp'] += 1
            
            # Les objets GT non matchés sont des faux négatifs
            for gt_idx, gt_obj in enumerate(gt_objects):
                if not gt_matched[gt_idx]:
                    total_fn += 1
                    line_metrics[gt_obj['ligne']]['fn'] += 1
        
        # Calculer les métriques finales
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Métriques par ligne
        line_results = {}
        for ligne in sorted(line_metrics.keys()):
            metrics = line_metrics[ligne]
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            
            line_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            line_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            line_f1 = 2 * line_precision * line_recall / (line_precision + line_recall) if (line_precision + line_recall) > 0 else 0
            
            line_results[ligne] = {
                'precision': line_precision,
                'recall': line_recall,
                'f1_score': line_f1,
                'support': tp + fn,  # Nombre réel d'objets de cette ligne
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        return {
            'global': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'total_detections': total_tp + total_fp,
                'total_ground_truth': total_tp + total_fn
            },
            'by_line': line_results
        }
    
    def print_detailed_report(self):
        """Affiche un rapport détaillé des métriques"""
        print("\n" + "="*60)
        print("RAPPORT DÉTAILLÉ DES MÉTRIQUES")
        print("="*60)
        
        if not self.metrics:
            print("❌ Aucune métrique disponible")
            return
        
        global_metrics = self.metrics['global']
        
        print("\n🎯 MÉTRIQUES GLOBALES:")
        print("-" * 30)
        print(f"Précision (Precision): {global_metrics['precision']:.3f}")
        print(f"Rappel (Recall):       {global_metrics['recall']:.3f}")
        print(f"F1-Score:              {global_metrics['f1_score']:.3f}")
        print(f"")
        print(f"Vrais Positifs (TP):   {global_metrics['total_tp']}")
        print(f"Faux Positifs (FP):    {global_metrics['total_fp']}")
        print(f"Faux Négatifs (FN):    {global_metrics['total_fn']}")
        print(f"")
        print(f"Total détections:      {global_metrics['total_detections']}")
        print(f"Total vérité terrain:  {global_metrics['total_ground_truth']}")
        
        print("\n📊 MÉTRIQUES PAR LIGNE:")
        print("-" * 50)
        print(f"{'Ligne':<6} {'Prec.':<6} {'Rapp.':<6} {'F1':<6} {'Supp.':<6} {'TP':<4} {'FP':<4} {'FN':<4}")
        print("-" * 50)
        
        for ligne in sorted(self.metrics['by_line'].keys()):
            metrics = self.metrics['by_line'][ligne]
            print(f"{ligne:<6} {metrics['precision']:<6.3f} {metrics['recall']:<6.3f} "
                  f"{metrics['f1_score']:<6.3f} {metrics['support']:<6} "
                  f"{metrics['tp']:<4} {metrics['fp']:<4} {metrics['fn']:<4}")
        
        # Moyennes pondérées
        total_support = sum(m['support'] for m in self.metrics['by_line'].values())
        if total_support > 0:
            weighted_precision = sum(m['precision'] * m['support'] for m in self.metrics['by_line'].values()) / total_support
            weighted_recall = sum(m['recall'] * m['support'] for m in self.metrics['by_line'].values()) / total_support
            weighted_f1 = sum(m['f1_score'] * m['support'] for m in self.metrics['by_line'].values()) / total_support
            
            print("-" * 50)
            print(f"{'Moy.P':<6} {weighted_precision:<6.3f} {weighted_recall:<6.3f} "
                  f"{weighted_f1:<6.3f} {total_support:<6}")
    
    def save_metrics_report(self, filename='metrics_report.json'):
        """Sauvegarde les métriques dans un fichier JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"📁 Métriques sauvegardées dans '{filename}'")
    
    def plot_metrics_visualization(self):
        """Génère des graphiques de visualisation des métriques"""
        if not self.metrics or not self.metrics['by_line']:
            print("❌ Pas de données à visualiser")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Données par ligne
        lignes = sorted(self.metrics['by_line'].keys())
        precisions = [self.metrics['by_line'][l]['precision'] for l in lignes]
        recalls = [self.metrics['by_line'][l]['recall'] for l in lignes]
        f1_scores = [self.metrics['by_line'][l]['f1_score'] for l in lignes]
        supports = [self.metrics['by_line'][l]['support'] for l in lignes]
        
        # 1. Précision par ligne
        bars1 = ax1.bar([f'L{l}' for l in lignes], precisions, alpha=0.7, color='skyblue')
        ax1.set_title('Précision par ligne')
        ax1.set_ylabel('Précision')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars1, precisions):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Rappel par ligne
        bars2 = ax2.bar([f'L{l}' for l in lignes], recalls, alpha=0.7, color='lightcoral')
        ax2.set_title('Rappel par ligne')
        ax2.set_ylabel('Rappel')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, recalls):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 3. F1-Score par ligne
        bars3 = ax3.bar([f'L{l}' for l in lignes], f1_scores, alpha=0.7, color='lightgreen')
        ax3.set_title('F1-Score par ligne')
        ax3.set_ylabel('F1-Score')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        for bar, val in zip(bars3, f1_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Support (nombre d'échantillons) par ligne
        bars4 = ax4.bar([f'L{l}' for l in lignes], supports, alpha=0.7, color='gold')
        ax4.set_title('Support (nombre d\'échantillons) par ligne')
        ax4.set_ylabel('Nombre d\'échantillons')
        ax4.grid(True, alpha=0.3)
        
        for bar, val in zip(bars4, supports):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('metrics_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualisations sauvegardées dans 'metrics_visualization.png'")


def main():
    """Fonction principale pour analyser les métriques"""
    print("📈 ANALYSEUR DE MÉTRIQUES - CHALLENGE MÉTRO")
    print("="*60)
    
    # Configuration
    gt_file = 'GTCHALLENGETEST.mat'
    results_file = 'teamsNN.mat'
    resize_factor = 1
    
    # Vérifier que les fichiers existent
    import os
    if not os.path.exists(gt_file):
        print(f"❌ Fichier de vérité terrain '{gt_file}' introuvable")
        return
    
    if not os.path.exists(results_file):
        print(f"❌ Fichier de résultats '{results_file}' introuvable")
        print("   Exécutez d'abord le script de challenge pour générer les résultats")
        return
    
    # Créer l'analyseur
    analyzer = MetricsAnalyzer(gt_file, results_file, resize_factor)
    
    # Afficher le rapport détaillé
    analyzer.print_detailed_report()
    
    # Sauvegarder les métriques
    analyzer.save_metrics_report()
    
    # Générer les visualisations
    analyzer.plot_metrics_visualization()
    
    # Comparaison avec l'exemple de référence si disponible
    if os.path.exists('teamsEX.mat'):
        print("\n🔍 COMPARAISON AVEC L'EXEMPLE DE RÉFÉRENCE:")
        print("-" * 50)
        
        ref_analyzer = MetricsAnalyzer(gt_file, 'teamsEX.mat', 0.5)  # L'exemple utilise resize_factor=0.5
        
        # Métriques de référence
        ref_global = ref_analyzer.metrics['global']
        your_global = analyzer.metrics['global']
        
        print(f"{'Métrique':<15} {'Votre système':<15} {'Référence':<15} {'Différence':<15}")
        print("-" * 60)
        print(f"{'Précision':<15} {your_global['precision']:<15.3f} {ref_global['precision']:<15.3f} {your_global['precision']-ref_global['precision']:<15.3f}")
        print(f"{'Rappel':<15} {your_global['recall']:<15.3f} {ref_global['recall']:<15.3f} {your_global['recall']-ref_global['recall']:<15.3f}")
        print(f"{'F1-Score':<15} {your_global['f1_score']:<15.3f} {ref_global['f1_score']:<15.3f} {your_global['f1_score']-ref_global['f1_score']:<15.3f}")
        
        # Déterminer les points forts et faibles
        print(f"\n💡 ANALYSE COMPARATIVE:")
        if your_global['f1_score'] > ref_global['f1_score']:
            print(f"   ✅ Votre système surpasse la référence (Δ F1: +{your_global['f1_score']-ref_global['f1_score']:.3f})")
        elif your_global['f1_score'] > ref_global['f1_score'] - 0.05:
            print(f"   ⚖️ Performance similaire à la référence (Δ F1: {your_global['f1_score']-ref_global['f1_score']:.3f})")
        else:
            print(f"   🔧 Marge d'amélioration (Δ F1: {your_global['f1_score']-ref_global['f1_score']:.3f})")
        
        if your_global['precision'] > ref_global['precision']:
            print(f"   🎯 Meilleure précision (+{your_global['precision']-ref_global['precision']:.3f})")
        
        if your_global['recall'] > ref_global['recall']:
            print(f"   🔍 Meilleur rappel (+{your_global['recall']-ref_global['recall']:.3f})")
    
    print(f"\n🎉 ANALYSE TERMINÉE!")
    print(f"📁 Fichiers générés:")
    print(f"   • metrics_report.json - Métriques détaillées")
    print(f"   • metrics_visualization.png - Graphiques")


if __name__ == "__main__":
    main()