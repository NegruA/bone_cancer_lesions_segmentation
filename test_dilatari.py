import utils
import matplotlib.pylab as plt
import pickle
import cv2
from scipy.ndimage import binary_fill_holes
import numpy as np
from skimage.measure import label, regionprops
from skimage import io, filters, color, measure
from skimage.morphology import erosion, square, flood_fill, closing, dilation, disk, diamond
from skimage.filters import threshold_otsu, median, gaussian
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import time

with open('D:\Facultate\Licenta\Codul sursa\obiecte\img_boala\img_boala.pkl', 'rb') as f:
    source_imgs = pickle.load(f)
    
source_imgs =source_imgs[::-1]
imgs_index = [33, 36, 37, 38, 39, 41,42,44,49, 50, 51,57,58,59,60, 156, 158, 159, 160, 205, 237]
# imgs_index = [41,42,44,49, 50]
# imgs_index = [156, 158, 159, 160, 205, 237]

imgs = [ source_imgs[i] for i in imgs_index]
imgs = [ utils.crop_image(img, 210, 200, 200, 170)for img in imgs]
segmentations = [ utils.manual_filter(img, 170, 200).astype(np.uint8) for img in imgs]
masks = []

kernel = np.ones((7, 7), np.uint8)

for segmentation in segmentations :
    
    plt.imshow(segmentation, cmap = 'gray')
    plt.colorbar()
    plt.show()
    
    masca_dilatata = dilation(segmentation, disk(3))
    
    plt.imshow(masca_dilatata, cmap = 'gray')
    plt.colorbar()
    plt.show()
    #masca_inchisa = cv2.morphologyEx(masca_dilatata, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
    
    # masca_finala = binary_fill_holes(masca_inchisa).astype(np.uint8)
    masca_finala = erosion(masca_dilatata, disk(3))
    
        
    plt.imshow(masca_dilatata, cmap = 'gray')
    plt.colorbar()
    plt.show()
    
    plt.imshow(masca_finala, cmap = 'gray')
    plt.colorbar()
    plt.show()
    
    masks.append(masca_finala)
    
    print('--------------------------------')


for i in range(len(masks)):
        fig, axs = plt.subplots(1, 2, figsize=(12,12))
        
        axs[0].imshow(imgs[i], cmap='gray', vmin = 0, vmax = 255)
        
        axs[1].imshow(masks[i], cmap='gray', vmin = 0, vmax = 255)

        plt.show()
        