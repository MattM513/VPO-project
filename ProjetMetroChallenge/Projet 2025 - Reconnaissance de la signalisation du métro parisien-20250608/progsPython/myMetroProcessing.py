# myMetroProcessing.py
import numpy as np
from ultralytics import YOLO
import os

class FinalMetroSystem:
    """
    Classe qui encapsule le modèle YOLO entraîné.
    """
    def __init__(self, model_path):
        """
        Charge le modèle YOLO au moment de l'initialisation.
        """
        # Vérifie si le modèle existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier modèle n'a pas été trouvé à l'emplacement : {model_path}")
        self.model = YOLO(model_path)
        
        # On peut définir des seuils ici si besoin
        self.confidence_threshold = 0.2 # Seuil de confiance pour garder une détection

    def predict(self, image_np):
        """
        Effectue une prédiction sur une image numpy.
        L'image doit être au format RGB, avec des valeurs de 0 à 255.
        """
        # YOLO prend en charge les images numpy directement
        results = self.model(image_np, verbose=False) # verbose=False pour ne pas polluer la console
        
        # Le résultat est une liste (même pour une seule image)
        return results[0]

# La fonction de traitement d'une image doit être modifiée pour utiliser notre système.
# Elle sera appelée par metro2025_ID.py
def processOneMetroImage(image_name, image_np_01, image_num, resize_factor, save_images=False, metro_system=None):
    """
    Traite une image pour détecter et reconnaître les lignes de métro.
    """
    if metro_system is None:
        raise ValueError("Le système de métro (modèle YOLO) n'a pas été fourni.")

    # L'image est fournie en float 0-1. On la convertit en uint8 0-255 pour le modèle.
    image_np_uint8 = (image_np_01 * 255).astype(np.uint8)

    # 1. Faire la prédiction avec YOLO
    results = metro_system.predict(image_np_uint8)

    # 2. Formater les résultats dans le format attendu
    # Format de sortie : [img_num, y1, y2, x1, x2, class_label]
    detections = []
    for r in results.boxes:
        # Filtrer par confiance
        if r.conf[0] >= metro_system.confidence_threshold:
            # Coordonnées de la boite
            x1, y1, x2, y2 = r.xyxy[0].cpu().numpy().astype(int)
            
            # Classe prédite (ID de classe 0-13)
            class_id = int(r.cls[0].cpu().numpy())
            
            # Convertir l'ID de classe en numéro de ligne (0 -> 1, 1 -> 2, etc.)
            class_label = class_id + 1
            
            # Ajouter au format de sortie attendu
            # Attention au resize_factor si vous l'utilisez ! Ici on assume 1.
            detections.append([image_num, y1, y2, x1, x2, class_label])
    
    if not detections:
        # Si aucune détection, retourner un array vide avec la bonne forme
        return image_np_01, np.empty((0, 6))

    return image_np_01, np.array(detections)