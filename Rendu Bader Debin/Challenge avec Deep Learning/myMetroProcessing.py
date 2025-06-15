# -*- coding: utf-8 -*-

import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage as ski
import os

class FinalMetroSystem:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = 0.4

    def predict(self, img):
        return self.model(img, verbose=False)[0]


def processOneMetroImage(nom, im, n, resizeFactor, metro_system=None):
    if resizeFactor != 1:
        im_resized = ski.transform.resize(im, (int(im.shape[0] * resizeFactor), int(im.shape[1] * resizeFactor)),
                                          anti_aliasing=True, preserve_range=True).astype(im.dtype)
    else:
        im_resized = im

    if metro_system is None:
        raise ValueError("metro_system doit être fourni")

    im_uint8 = (im_resized * 255).astype(np.uint8)
    results = metro_system.predict(im_uint8)

    bd = []
    for r in results.boxes:
        if r.conf[0] >= metro_system.conf_threshold:
            x1, y1, x2, y2 = r.xyxy[0].cpu().numpy().astype(int)
            cls = int(r.cls[0].cpu().numpy()) + 1
            bd.append([n, y1, y2, x1, x2, cls])

    if not bd:
        bd = np.empty((0, 6))
    else:
        bd = np.array(bd)

    return im_resized, bd


def draw_rectangle(x1, x2, y1, y2, color):
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
