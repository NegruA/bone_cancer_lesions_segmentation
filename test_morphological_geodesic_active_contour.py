import utils 
import matplotlib.pylab as plt
from scipy.ndimage import binary_fill_holes
import SimpleITK as sitk
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
import time
from sklearn.metrics import f1_score, confusion_matrix
from skimage import feature
import numpy as np
from skimage import filters, io, color
from skimage.morphology import closing, square

def compute_metrics(truth, prediction):

    truth = np.array(truth).flatten()
    prediction = np.array(prediction).flatten()
    
    conf_matrix = confusion_matrix(truth, prediction)
    
    if conf_matrix.shape == (1, 1):

        value = conf_matrix[0, 0]
        if 0 in truth and 0 in prediction:  
            TN = value
            FP, FN, TP = 0, 0, 0
        elif 1 in truth and 1 in prediction:  
            TP = value
            TN, FP, FN = 0, 0, 0
    else:
        TN, FP, FN, TP = conf_matrix.ravel()

    total_points = TP + TN + FP + FN
    acuratete = (TP + TN) / total_points if total_points != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    precizie = TP / (TP + FP) if TP + FP != 0 else 0
    f1 = f1_score(truth, prediction, zero_division=1)
    
    return acuratete, precizie, recall, f1, TN, FP, FN, TP


directory = r'D:\Facultate\Licenta\Baza de imagini\manifest-1669911955156\CMB-MML\CMB-MML-MSB-02286\01-01-1960-NA-CT AbdPelvis-16734\5.000000-Body Sagittal 3.0 VENOUSPHASESagittal CE-34049'

segmentation_test_dir = r'D:\Facultate\Licenta\Baza de imagini\manifest-1669911955156\CMB-MML\CMB-MML-MSB-02286\01-01-1960-NA-CT AbdPelvis-16734\5.000000-Body Sagittal 3.0 VENOUSPHASESagittal CE-34049\Segmentation.nrrd'

image = sitk.ReadImage(segmentation_test_dir)

truth_array = sitk.GetArrayFromImage(image)
      
imgs = utils.load_scan(directory)

start_time = time.time() 

acuratete_mgac = []
f1_mgac = []
tn_mgac = []
fp_mgac = []
fn_mgac = []
tp_mgac = []
precizia_mgac = []
recall_mgac = []
nr_pixeli = imgs[0].shape[0]*imgs[0].shape[1]

masked_imgs = []

for i, img in enumerate(imgs):
    print(i)
    init_ls = feature.canny(img, sigma=3)
    init_ls = closing(init_ls, square(11))
    
    plt.imshow(init_ls, cmap = 'gray')
    plt.colorbar()
    plt.show()
    
        
    # gimage = inverse_gaussian_gradient(img)
    # ls = morphological_geodesic_active_contour(gimage, 230, init_ls,
    #                                         smoothing=2, balloon=0,
    #                                         threshold=0.1)
    
    # masca_finala = ls
    
    # plt.imshow(masca_finala, cmap = 'gray')
    # plt.colorbar()
    # plt.show()
    
    # conf_matrix = confusion_matrix(truth_array[i].flatten(), masca_finala.flatten())
    # f1 = f1_score(truth_array[i].flatten(), masca_finala.flatten()) 
    
    # print("aici e aia" + str(conf_matrix))
    
    # acuratete, precizie, recall, f1, TN, FP, FN, TP = compute_metrics(truth_array[i].flatten(), masca_finala.flatten())
    
    # acuratete_mgac.append(acuratete)
    # f1_mgac.append(f1)
    # recall_mgac.append(recall)
    # precizia_mgac.append(precizie)
    # tn_mgac.append(TN/nr_pixeli)
    # fp_mgac.append(FP/nr_pixeli)
    # fn_mgac.append(FN/nr_pixeli)
    # tp_mgac.append(TP/nr_pixeli)
    

end_time = time.time() 
elapsed_time = end_time - start_time

num_images = len(imgs)    
    
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(range(num_images), tn_mgac, marker='o', label='Adevarat negativ (AN)')
ax.plot(range(num_images), fp_mgac, marker='x', color='r', label='Fals pozitiv (FP)')
ax.plot(range(num_images), fn_mgac, marker='s', color='g', label='Fals negativ (FN)')
ax.plot(range(num_images), tp_mgac, marker='d', color='m', label='Adevarat pozitiv (AP)')

ax.set_title('Metrica performantei segmentarii mgac in lotul de imagini')
ax.set_xlabel('Numarul imaginii')
ax.set_ylabel('Rata de clasificare')

ax.legend()

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

ax1.plot(range(num_images), fp_mgac, marker='x', color='r', label='Fals pozitiv (FP)')
ax1.plot(range(num_images), tp_mgac, marker='d', color='m', label='Adevarat pozitiv (AP)')
ax1.set_title('Fals si adevarat pozitiv pentru metoda mgac')
ax1.set_xlabel('Numarul imaginii')
ax1.set_ylabel('Rata de clasificare')
ax1.legend()

ax2.plot(range(num_images), tn_mgac, marker='o', label='Adevarat negativ (AN)')
ax2.plot(range(num_images), fn_mgac, marker='s', color='g', label='Fals negativ (FN)')
ax2.set_title('Fals si adevarat negativ pentru metoda mgac')
ax2.set_xlabel('Numarul imaginii')
ax2.set_ylabel('Rata de clasificare')
ax2.legend()

plt.show()

text = [f'Img{i}' for i in range(num_images)]

media_acuratete = np.mean(acuratete_mgac)
media_precizie = np.mean(precizia_mgac)
media_rapel = np.mean(recall_mgac)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.barh(text, acuratete_mgac, color='purple')
plt.axvline(x=media_acuratete, color='r', linestyle='--', label=f'Medie: {media_acuratete:.2f}')
plt.xlabel('Acuratețe')
plt.ylabel('Poze')
plt.title('Acuratețea segmentarii mgac pentru fiecare poza')
plt.legend()
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])

plt.subplot(1, 3, 2)
plt.barh(text, precizia_mgac, color='blue')
plt.axvline(x=media_precizie, color='r', linestyle='--', label=f'Medie: {media_precizie:.2f}')
plt.xlabel('Precizie')
plt.ylabel('Poze')
plt.title('Precizia segmentarii mgac pentru fiecare poza')
plt.legend()
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])

plt.subplot(1, 3, 3)
plt.barh(text, recall_mgac, color='green')
plt.axvline(x=media_rapel, color='r', linestyle='--', label=f'Medie: {media_rapel:.2f}')
plt.xlabel('Rapel')
plt.ylabel('Poze')
plt.title('Rapelul segmentarii mgac pentru fiecare poza')
plt.legend()
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])

plt.tight_layout()
plt.show()


# plt.imshow(img, cmap = 'gray')
# plt.colorbar()
# plt.show()

# plt.imshow(init_ls, cmap = 'gray')
# plt.colorbar()
# plt.show()

# plt.imshow(gimage, cmap = 'gray')
# plt.colorbar()
# plt.show()

# plt.imshow(ls, cmap = 'gray')
# plt.colorbar()
# plt.show()

# # plt.imshow(masked_img, cmap = 'gray')
# # plt.colorbar()
# # plt.show()
 
                