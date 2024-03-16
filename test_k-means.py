import SimpleITK as sitk
import matplotlib.pylab as plt
import utils
from sklearn.cluster import KMeans
import numpy as np 
from skimage.morphology import erosion, dilation, disk
from sklearn.metrics import f1_score, confusion_matrix

directory = r'D:\Facultate\Licenta\Baza de imagini\manifest-1669911955156\CMB-MML\CMB-MML-MSB-02286\01-01-1960-NA-CT AbdPelvis-16734\5.000000-Body Sagittal 3.0 VENOUSPHASESagittal CE-34049'

segmentation_test_dir = r'D:\Facultate\Licenta\Baza de imagini\manifest-1669911955156\CMB-MML\CMB-MML-MSB-02286\01-01-1960-NA-CT AbdPelvis-16734\5.000000-Body Sagittal 3.0 VENOUSPHASESagittal CE-34049\Segmentation.nrrd'

image = sitk.ReadImage(segmentation_test_dir)

truth_array = sitk.GetArrayFromImage(image)

imgs = utils.load_scan(directory)

rezultate_segmentare = []
centroizi_init = [0.415512, 151.183, 216.824 , 94.0981]
centroizi_init = np.array(centroizi_init).reshape(-1, 1)

kernel = np.ones((11, 11), np.uint8)         

acuratete_kmeans = []
f1_kmeans = []
f1_kmeans = []
tn_kmeans = []
fp_kmeans = []
fn_kmeans = []
tp_kmeans = []
precizia_kmeans = []
recall_kmeans = []    
nr_pixeli = imgs[0].shape[0]*imgs[0].shape[1]

for i, img in enumerate(imgs):
    
    pixels = img.reshape(-1, 1)

    kmeans = KMeans(n_clusters=len(centroizi_init), init=np.array(centroizi_init), n_init=1, random_state=42)
    etichete = kmeans.fit_predict(pixels)

    segmentare = etichete.reshape(img.shape)
    
    masca_osos = (segmentare == 2).astype(np.uint8)
    
    masca_dilatata = dilation(masca_osos, disk(5))   
    
    #masca_inchisa = cv2.morphologyEx(masca_dilatata, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
    
    masca_finala = erosion(masca_dilatata, disk(3)).astype(np.uint8)
    
    rezultate_segmentare.append(masca_finala)
    
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # axs[0, 0].imshow(segmentare, cmap='nipy_spectral')
    # axs[0, 0].set_title('Segmentare K-means')
    
    # axs[0, 1].imshow(masca_osos, cmap='gray')
    # axs[0, 1].set_title('Masca preliminara os')
    
    # axs[1, 0].imshow(masca_dilatata, cmap='gray')
    # axs[1, 0].set_title('Masca dilatata')
    
    # axs[1, 1].imshow(masca_finala, cmap='gray')
    # axs[1, 1].set_title('Masca erodata/ finala')
    
    # for ax in axs.flat:
    #     ax.axis('off')
    
    # plt.show()
    
    conf_matrix = confusion_matrix(truth_array[i].flatten(), masca_finala.flatten())
    f1 = f1_score(truth_array[i].flatten(), masca_finala.flatten()) 
    
    TN, FP, FN, TP = conf_matrix.ravel()
    
    total_points = TP + TN + FP + FN
    if total_points == 0:
        acuratete = 0 
    else:
        acuratete = (TP + TN) / total_points
    
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    
    if TP + FP != 0:
        precizie = TP / (TP + FP)
    else:
        precizie = 0
    
    acuratete_kmeans.append(acuratete)
    f1_kmeans.append(f1)
    recall_kmeans.append(recall)
    precizia_kmeans.append(precizie)
    tn_kmeans.append(TN/nr_pixeli)
    fp_kmeans.append(FP/nr_pixeli)
    fn_kmeans.append(FN/nr_pixeli)
    tp_kmeans.append(TP/nr_pixeli)

num_images = len(imgs)    
    
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(range(num_images), tn_kmeans, marker='o', label='Adevarat negativ (AN)')
ax.plot(range(num_images), fp_kmeans, marker='x', color='r', label='Fals pozitiv (FP)')
ax.plot(range(num_images), fn_kmeans, marker='s', color='g', label='Fals negativ (FN)')
ax.plot(range(num_images), tp_kmeans, marker='d', color='m', label='Adevarat pozitiv (AP)')

ax.set_title('Metrica performantei segmentarii kmeans in lotul de imagini')
ax.set_xlabel('Numarul imaginii')
ax.set_ylabel('Rata de clasificare')

ax.legend()

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

ax1.plot(range(num_images), fp_kmeans, marker='x', color='r', label='Fals pozitiv (FP)')
ax1.plot(range(num_images), tp_kmeans, marker='d', color='m', label='Adevarat pozitiv (AP)')
ax1.set_title('Fals si adevarat pozitiv pentru metoda kmeans')
ax1.set_xlabel('Numarul imaginii')
ax1.set_ylabel('Rata de clasificare')
ax1.legend()

ax2.plot(range(num_images), tn_kmeans, marker='o', label='Adevarat negativ (AN)')
ax2.plot(range(num_images), fn_kmeans, marker='s', color='g', label='Fals negativ (FN)')
ax2.set_title('Fals si adevarat negativ pentru metoda kmeans')
ax2.set_xlabel('Numarul imaginii')
ax2.set_ylabel('Rata de clasificare')
ax2.legend()

plt.show()

text = [f'Img{i}' for i in range(num_images)]

media_acuratete = np.mean(acuratete_kmeans)
media_precizie = np.mean(precizia_kmeans)
media_rapel = np.mean(recall_kmeans)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.barh(text, acuratete_kmeans, color='purple')
plt.axvline(x=media_acuratete, color='r', linestyle='--', label=f'Medie: {media_acuratete:.2f}')
plt.xlabel('Acuratețe')
plt.ylabel('Poze')
plt.title('Acuratețea segmentarii kmeans pentru fiecare poza')
plt.legend()
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])

plt.subplot(1, 3, 2)
plt.barh(text, precizia_kmeans, color='blue')
plt.axvline(x=media_precizie, color='r', linestyle='--', label=f'Medie: {media_precizie:.2f}')
plt.xlabel('Precizie')
plt.ylabel('Poze')
plt.title('Precizia segmentarii kmeans pentru fiecare poza')
plt.legend()
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])

plt.subplot(1, 3, 3)
plt.barh(text, recall_kmeans, color='green')
plt.axvline(x=media_rapel, color='r', linestyle='--', label=f'Medie: {media_rapel:.2f}')
plt.xlabel('Rapel')
plt.ylabel('Poze')
plt.title('Rapelul segmentarii kmeans pentru fiecare poza')
plt.legend()
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])

plt.tight_layout()
plt.show()