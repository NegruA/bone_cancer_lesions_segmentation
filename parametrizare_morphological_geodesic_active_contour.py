from skimage import measure
from sklearn.cluster import KMeans
import utils 
import numpy as np 
import matplotlib.pylab as plt
import cv2
import SimpleITK as sitk
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from sklearn.metrics import f1_score
from skimage import feature
from skimage.morphology import closing, square

def objective(image, init_level_set, params, ground_truth):
    
    gimage = inverse_gaussian_gradient(image)
    
    ls = morphological_geodesic_active_contour(gimage, 250, init_level_set,
                                               smoothing=params['smoothing'],
                                               balloon=params['balloon'],
                                               threshold=params['threshold'])

    score = f1_score(ground_truth.flatten(), ls.flatten())
    # print(score)
    # plt.imshow(ground_truth, cmap = 'gray')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(ls, cmap = 'gray')
    # plt.colorbar()
    # plt.show()    
    return score, ls

def grid_search(image, init_level_set, param_grid, ground_truth):
    best_score = -1
    best_params = None

    for smoothing in param_grid['smoothing']:
        print('SMOOTING:', smoothing)
        for balloon in param_grid['balloon']:
            print('balloon', balloon)
            for threshold in param_grid['threshold']:
                print('threshold', threshold)
                params = {'smoothing': smoothing, 'balloon': balloon, 'threshold': threshold}
                score, ls = objective(image, init_level_set, params, ground_truth)
                if score > best_score:
                    print('---------------------')
                    print("S-a gasit un scor mai bun decat ", str(best_score), '. Noul scor mai bun este ', str(score))
                    print("Parametrii noi sunt:", params)

                    plt.imshow(ls, cmap = 'gray')
                    plt.colorbar()
                    plt.show()
                    print('---------------------')
                    best_score = score
                    best_params = params

    return best_params, best_score

directory = r'D:\Facultate\Licenta\Baza de imagini\manifest-1669911955156\CMB-MML\CMB-MML-MSB-02286\01-01-1960-NA-CT AbdPelvis-16734\5.000000-Body Sagittal 3.0 VENOUSPHASESagittal CE-34049'

segmentation_test_dir = r'D:\Facultate\Licenta\Baza de imagini\manifest-1669911955156\CMB-MML\CMB-MML-MSB-02286\01-01-1960-NA-CT AbdPelvis-16734\5.000000-Body Sagittal 3.0 VENOUSPHASESagittal CE-34049\Segmentation.nrrd'

image = sitk.ReadImage(segmentation_test_dir)

array = sitk.GetArrayFromImage(image)
      
imgs = utils.load_scan(directory)

i = 60
image = imgs [i]
init_level_set = feature.canny(imgs[i], sigma=3)
init_level_set = closing(init_level_set, square(11))
ground_truth = array[i]

param_grid = {
    'smoothing': [1, 2, 3, 4, 5],
    'balloon': [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75],
    'threshold': [0.1, 0.2, 0.3]
}

score = grid_search(image, init_level_set, param_grid, ground_truth)