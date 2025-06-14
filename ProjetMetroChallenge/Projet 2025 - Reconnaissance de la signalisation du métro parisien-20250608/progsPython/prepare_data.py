# prepare_data.py
import os
import scipy.io as sio
import numpy as np
from PIL import Image

def convert_to_yolo_format(mat_file_path, images_dir_path, output_dir):
    # Charger les données .mat
    data = sio.loadmat(mat_file_path)['BD']

    # Créer les dossiers de sortie
    train_img_path = os.path.join(output_dir, 'images', 'train')
    # Pour la validation, nous pourrions utiliser une partie des données d'apprentissage
    # ou les données de test si elles sont annotées. Ici, on met tout en train.
    train_lbl_path = os.path.join(output_dir, 'labels', 'train')

    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(train_lbl_path, exist_ok=True)

    # Dictionnaire pour stocker les annotations par image
    annotations = {}

    for row in data:
        img_num = int(row[0])
        # Les coordonnées sont [y1, y2, x1, x2]
        y1, y2, x1, x2 = row[1:5]
        # La classe est le numéro de ligne. On va mapper 1->0, 2->1, ..., 14->13
        class_id = int(row[5]) - 1

        if img_num not in annotations:
            annotations[img_num] = []
        annotations[img_num].append((class_id, x1, y1, x2, y2))

    # Traiter chaque image
    for img_num, bboxes in annotations.items():
        img_name = f'IM ({img_num}).JPG'
        src_img_path = os.path.join(images_dir_path, img_name)
        
        if not os.path.exists(src_img_path):
            print(f"Attention : Image {src_img_path} non trouvée.")
            continue
        
        # Obtenir les dimensions de l'image
        with Image.open(src_img_path) as img:
            img_w, img_h = img.size

        # Copier l'image vers le dossier de destination
        os.system(f'copy "{src_img_path}" "{train_img_path}"')

        # Créer le fichier de label .txt
        label_file_path = os.path.join(train_lbl_path, f'IM ({img_num}).txt')
        with open(label_file_path, 'w') as f:
            for bbox in bboxes:
                class_id, x1, y1, x2, y2 = bbox
                
                # Convertir en format YOLO (centre_x, centre_y, largeur, hauteur) normalisé
                box_w = x2 - x1
                box_h = y2 - y1
                x_center = x1 + box_w / 2
                y_center = y1 + box_h / 2

                # Normaliser
                x_center_norm = x_center / img_w
                y_center_norm = y_center / img_h
                width_norm = box_w / img_w
                height_norm = box_h / img_h

                f.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

    print(f"Conversion terminée. Données YOLO créées dans le dossier '{output_dir}'")

# Lancer la conversion
convert_to_yolo_format('Apprentissage.mat', '../BD_METRO', 'Metro_Dataset')