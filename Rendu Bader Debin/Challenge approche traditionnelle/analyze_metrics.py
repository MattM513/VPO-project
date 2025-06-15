# -*- coding: utf-8 -*-
"""
Analyseur de métriques pour le challenge métro
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from collections import defaultdict
import json
import os

class MetricsAnalyzer:
    def __init__(self, gt_file, results_file, resize_factor=1):
        self.gt_file = gt_file
        self.results_file = results_file
        self.resize_factor = resize_factor
        self.gt_data = self._load_ground_truth()
        self.results_data = self._load_results()
        self.metrics = self._calculate_detailed_metrics()

    def _load_ground_truth(self):
        try:
            gt_mat = sio.loadmat(self.gt_file)
            return gt_mat['BD']
        except Exception as e:
            print(f"Erreur lors du chargement de {self.gt_file}: {e}")
            return np.array([])

    def _load_results(self):
        try:
            results_mat = sio.loadmat(self.results_file)
            return results_mat['BD']
        except Exception as e:
            print(f"Erreur lors du chargement de {self.results_file}: {e}")
            return np.array([])

    def _calculate_iou(self, box1, box2):
        x1_1, x2_1, y1_1, y2_1 = box1
        x1_2, x2_2, y1_2, y2_2 = box2
        x1_inter = max(x1_1, x1_2)
        x2_inter = min(x2_1, x2_2)
        y1_inter = max(y1_1, y1_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _calculate_detailed_metrics(self):
        if len(self.gt_data) == 0 or len(self.results_data) == 0:
            print("Données insuffisantes pour calculer les métriques")
            return {}

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

        total_tp = total_fp = total_fn = 0
        line_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        iou_threshold = 0.5

        for img_num in set(list(gt_by_image.keys()) + list(results_by_image.keys())):
            gt_objects = gt_by_image.get(img_num, [])
            detected_objects = results_by_image.get(img_num, [])
            gt_matched = [False] * len(gt_objects)

            for detection in detected_objects:
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt_obj in enumerate(gt_objects):
                    if gt_matched[gt_idx]:
                        continue
                    iou = self._calculate_iou(detection['bbox'], gt_obj['bbox'])
                    if (iou > best_iou and iou >= iou_threshold and detection['ligne'] == gt_obj['ligne']):
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0:
                    total_tp += 1
                    line_metrics[detection['ligne']]['tp'] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    total_fp += 1
                    line_metrics[detection['ligne']]['fp'] += 1

            for gt_idx, gt_obj in enumerate(gt_objects):
                if not gt_matched[gt_idx]:
                    total_fn += 1
                    line_metrics[gt_obj['ligne']]['fn'] += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

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
                'support': tp + fn,
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
        print("\n" + "="*60)
        print("RAPPORT DÉTAILLÉ DES MÉTRIQUES")
        print("="*60)

        if not self.metrics:
            print("Aucune métrique disponible")
            return

        global_metrics = self.metrics['global']

        print("\nMÉTRIQUES GLOBALES:")
        print("-" * 30)
        print(f"Précision: {global_metrics['precision']:.3f}")
        print(f"Rappel:    {global_metrics['recall']:.3f}")
        print(f"F1-Score:  {global_metrics['f1_score']:.3f}")
        print(f"TP:        {global_metrics['total_tp']}")
        print(f"FP:        {global_metrics['total_fp']}")
        print(f"FN:        {global_metrics['total_fn']}")
        print(f"Détections: {global_metrics['total_detections']}")
        print(f"Vérité terrain: {global_metrics['total_ground_truth']}")

        print("\nMÉTRIQUES PAR LIGNE:")
        print("-" * 50)
        print(f"{'Ligne':<6} {'Prec.':<6} {'Rapp.':<6} {'F1':<6} {'Supp.':<6} {'TP':<4} {'FP':<4} {'FN':<4}")
        print("-" * 50)

        for ligne in sorted(self.metrics['by_line'].keys()):
            metrics = self.metrics['by_line'][ligne]
            print(f"{ligne:<6} {metrics['precision']:<6.3f} {metrics['recall']:<6.3f} "
                  f"{metrics['f1_score']:<6.3f} {metrics['support']:<6} "
                  f"{metrics['tp']:<4} {metrics['fp']:<4} {metrics['fn']:<4}")

    def save_metrics_report(self, filename='metrics_report.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"Métriques sauvegardées dans '{filename}'")

    def plot_metrics_visualization(self):
        if not self.metrics or not self.metrics['by_line']:
            print("Pas de données à visualiser")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        lignes = sorted(self.metrics['by_line'].keys())
        precisions = [self.metrics['by_line'][l]['precision'] for l in lignes]
        recalls = [self.metrics['by_line'][l]['recall'] for l in lignes]
        f1_scores = [self.metrics['by_line'][l]['f1_score'] for l in lignes]
        supports = [self.metrics['by_line'][l]['support'] for l in lignes]

        # Visualisation des métriques
        axes = [ax1, ax2, ax3, ax4]
        titles = ['Précision par ligne', 'Rappel par ligne', 'F1-Score par ligne', "Support par ligne"]
        datas = [precisions, recalls, f1_scores, supports]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        ylabels = ['Précision', 'Rappel', 'F1-Score', "Nombre"]

        for ax, title, data, color, ylabel in zip(axes, titles, datas, colors, ylabels):
            bars = ax.bar([f'L{l}' for l in lignes], data, alpha=0.7, color=color)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            for bar, val in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}' if isinstance(val, float) else f'{val}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig('metrics_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Visualisations sauvegardées dans 'metrics_visualization.png'")

def main():
    gt_file = 'GTCHALLENGETEST.mat'
    results_file = 'teamsNN.mat'
    resize_factor = 1

    if not os.path.exists(gt_file):
        print(f"Fichier de vérité terrain '{gt_file}' introuvable")
        return

    if not os.path.exists(results_file):
        print(f"Fichier de résultats '{results_file}' introuvable")
        return

    analyzer = MetricsAnalyzer(gt_file, results_file, resize_factor)
    analyzer.print_detailed_report()
    analyzer.save_metrics_report()
    analyzer.plot_metrics_visualization()

if __name__ == "__main__":
    main()
