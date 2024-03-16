import SimpleITK as sitk
import matplotlib.pylab as plt
import utils
import numpy as np 
from skimage.morphology import erosion, dilation, disk
from sklearn.metrics import f1_score, confusion_matrix

directory = r'D:\Facultate\Licenta\Baza de imagini\manifest-1669911955156\CMB-MML\CMB-MML-MSB-02286\01-01-1960-NA-CT AbdPelvis-16734\5.000000-Body Sagittal 3.0 VENOUSPHASESagittal CE-34049'

segmentation_test_dir = r'D:\Facultate\Licenta\Baza de imagini\manifest-1669911955156\CMB-MML\CMB-MML-MSB-02286\01-01-1960-NA-CT AbdPelvis-16734\5.000000-Body Sagittal 3.0 VENOUSPHASESagittal CE-34049\Segmentation.nrrd'

image = sitk.ReadImage(segmentation_test_dir)

truth_array = sitk.GetArrayFromImage(image)

imgs = utils.load_scan(directory)

masked_imgs = []
kernel = np.ones((9, 9), np.uint8)

acuratete_manual = []
f1_manual = []
tn_manual = []
fp_manual = []
fn_manual = []
tp_manual = []
precizia_manual = []
recall_manual = []    
nr_pixeli = imgs[0].shape[0]*imgs[0].shape[1]

# img = imgs[40]
for i,img in enumerate(imgs): 
    
    segmentare = utils.manual_filter(img, 174, 256 ).astype(np.uint8)
    
    
    masca_osos = (segmentare == 255).astype(np.uint8)
    
    masca_dilatata = dilation(masca_osos, disk(5))   
    
    #masca_inchisa = cv2.morphologyEx(masca_dilatata, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
    
    masca_finala = erosion(masca_dilatata, disk(3)).astype(np.uint8)
    
    masked_imgs.append(masca_finala)
    
    fig, axarr = plt.subplots(1, 3, figsize=(20, 5))
    
    # Afisarea diferitelor faze
    axarr[0].imshow(segmentare, cmap='gray')
    axarr[0].set_title('Segmentarea manuală preliminară')
    axarr[0].axis('off')
    
    axarr[1].imshow(masca_dilatata, cmap='gray')
    axarr[1].set_title('Masca dilatata')
    axarr[1].axis('off')
    
    axarr[2].imshow(masca_finala, cmap='gray')
    axarr[2].set_title('Masca erodata / finala')
    axarr[2].axis('off')
    plt.show()
    
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
    
    acuratete_manual.append(acuratete)
    f1_manual.append(f1)
    recall_manual.append(recall)
    precizia_manual.append(precizie)
    tn_manual.append(TN/nr_pixeli)
    fp_manual.append(FP/nr_pixeli)
    fn_manual.append(FN/nr_pixeli)
    tp_manual.append(TP/nr_pixeli)

num_images = len(imgs)    
    
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(range(num_images), tn_manual, marker='o', label='Adevarat negativ (AN)')
ax.plot(range(num_images), fp_manual, marker='x', color='r', label='Fals pozitiv (FP)')
ax.plot(range(num_images), fn_manual, marker='s', color='g', label='Fals negativ (FN)')
ax.plot(range(num_images), tp_manual, marker='d', color='m', label='Adevarat pozitiv (AP)')

ax.set_title('Metrica performantei segmentarii manuale in lotul de imagini')
ax.set_xlabel('Numarul imaginii')
ax.set_ylabel('Rata de clasificare')

ax.legend()

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

ax1.plot(range(num_images), fp_manual, marker='x', color='r', label='Fals pozitiv (FP)')
ax1.plot(range(num_images), tp_manual, marker='d', color='m', label='Adevarat pozitiv (AP)')
ax1.set_title('Fals si adevarat pozitiv pentru metoda manuala')
ax1.set_xlabel('Numarul imaginii')
ax1.set_ylabel('Rata de clasificare')
ax1.legend()

ax2.plot(range(num_images), tn_manual, marker='o', label='Adevarat negativ (AN)')
ax2.plot(range(num_images), fn_manual, marker='s', color='g', label='Fals negativ (FN)')
ax2.set_title('Fals si adevarat negativ pentru metoda manuala')
ax2.set_xlabel('Numarul imaginii')
ax2.set_ylabel('Rata de clasificare')
ax2.legend()

plt.show()

text = [f'Img{i}' for i in range(num_images)]

media_acuratete = np.mean(acuratete_manual)
media_precizie = np.mean(precizia_manual)
media_rapel = np.mean(recall_manual)

plt.figure(figsize=(18, 6))

# Grafic pentru Acuratețe
plt.subplot(1, 3, 1)
plt.barh(text, acuratete_manual, color='purple')
plt.axvline(x=media_acuratete, color='r', linestyle='--', label=f'Medie: {media_acuratete:.2f}')
plt.xlabel('Acuratețe')
plt.ylabel('Poze')
plt.title('Acuratețea segmentarii manuale pentru fiecare poza')
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])
plt.legend()

# Grafic pentru Precizie
plt.subplot(1, 3, 2)
plt.barh(text, precizia_manual, color='blue')
plt.axvline(x=media_precizie, color='r', linestyle='--', label=f'Medie: {media_precizie:.2f}')
plt.xlabel('Precizie')
plt.ylabel('Poze')
plt.title('Precizia segmentarii manuale pentru fiecare poza')
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])
plt.legend()

# Grafic pentru Rapel
plt.subplot(1, 3, 3)
plt.barh(text, recall_manual, color='green')
plt.axvline(x=media_rapel, color='r', linestyle='--', label=f'Medie: {media_rapel:.2f}')
plt.xlabel('Rapel')
plt.ylabel('Poze')
plt.title('Rapelul segmentarii manuale pentru fiecare poza')
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])
plt.legend()

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()

# masked_imgs2.append( masked_imgs1[0])
# score2.append( score1[0])

# for i in range(len(masked_imgs1)-2):
    
#     mask1 =  masked_imgs1[i]
#     mask2 =  masked_imgs1[i+1]
#     mask3 =  masked_imgs1[i+2]
    
#     new_mask = mask2.copy()
    
#     new_mask[(mask1 == 1) | (mask3 == 1)] = 1
    
#     masked_imgs2.append(new_mask)
    
#     score = f1_score(truth_array[i+1].flatten(), new_mask.flatten())
#     score2.append(score)
    
# masked_imgs2.append(masked_imgs1[len(masked_imgs1)-1])
# score2.append(score1[len(score1)-1])

# x_values = list(range(1, len(score1) + 1))

# plt.figure(figsize=(10, 6))
# plt.plot(x_values, score1, label="Score 1", marker='o', linestyle='-', color='blue')
# plt.plot(x_values, score2, label="Score 2", marker='o', linestyle='-', color='red')

# plt.title("Score Plots")
# plt.xlabel("Index")
# plt.ylabel("Score")
# plt.legend()

# plt.grid(True)
# plt.tight_layout()
# plt.show()

