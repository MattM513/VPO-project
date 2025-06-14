# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import scipy.io as sio

# Importer le système final
from myMetroProcessing import FinalMetroSystem, processOneMetroImage

# Variable globale pour le système entraîné
metro_system = None

def draw_rectangle(x1, x2, y1, y2, color):
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)

def metro2025(type_, viewImages=1):
    global metro_system
    
    plt.close('all')
    n = np.arange(1, 262)
    ok = True
    
    if type_ == 'Test':
        num_images = n[n % 3 != 0]
        BD = []
        
    elif type_ == 'Learn':
        num_images = n[n % 3 == 0]
        GT = sio.loadmat('Apprentissage.mat')['BD']
        BD = []
        
        # ENTRAÎNER LE SYSTÈME (une seule fois)
        if metro_system is None:
            print("🎓 CHARGEMENT DU SYSTÈME ENTRAÎNÉ...")
            model_path = '../../../runs/detect/train2/weights/best.pt' # Adaptez ce chemin !
            metro_system = FinalMetroSystem(model_path)
            print("✅ SYSTÈME CHARGÉ !")
        
    else:
        print("Bad identifier (should be 'Learn' or 'Test')")
        return None, None

    resize_factor = 1

    for n_val in num_images:
        nom = f'IM ({n_val})'
        im_path = os.path.join('../BD_METRO', f'{nom}.JPG')
        im = np.array(Image.open(im_path).convert('RGB')) / 255.0

        if viewImages:
            fig = plt.figure(figsize=(15,8))
            plt.subplot(1,2,1)
            plt.imshow(im)
    
        if type_ == 'Learn':
            ind = np.where(GT[:, 0] == n_val)[0]
            titre = 'GT: '
            for k in ind:
                bbox = np.round(resize_factor * GT[k, 1:5]).astype(int)
                if viewImages:
                    draw_rectangle(bbox[2], bbox[3], bbox[0], bbox[1], 'g')
                    titre += f'{int(GT[k, 5])}-'
            if viewImages:        
                plt.title(titre, fontsize=30)

        # Utiliser le système final entraîné
        im_resized, bd = processOneMetroImage(nom, im, n_val, resize_factor, 
                                            save_images=False, metro_system=metro_system)
            
        if viewImages:    
            plt.subplot(1,2,2)
            plt.imshow(im)
            titre = 'MyProg: '
            for k in range(bd.shape[0]):
                draw_rectangle(int(bd[k, 3]), int(bd[k, 4]), int(bd[k, 1]), int(bd[k, 2]), 'm')
                titre += f'{int(bd[k, 5])}-'
            plt.title(titre, fontsize=30)
            plt.show()
            plt.pause(0.1)

        BD.extend(bd.tolist())

    file_out = 'myResults.mat'
    sio.savemat(file_out, {'BD': np.array(BD)})

    return file_out, resize_factor

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        metro2025(mode, viewImages=1)
    else:
        print("Usage: python metro2025_ID.py [Learn|Test]")