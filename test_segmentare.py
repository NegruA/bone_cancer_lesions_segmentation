import utils
import matplotlib.pylab as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage import filters
from skimage.morphology import erosion, square, flood_fill, dilation, disk
from skimage import measure
from skimage import draw
import SimpleITK as sitk
from sklearn.metrics import f1_score, confusion_matrix

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

truth_array = []

for i in range(1,20):

    segmentation_test_dir = f'D:\Facultate\Licenta\Baza de imagini\ARCHIVE-RO01-04-20190813\dicoms\seg_boala\Segmentation_{i}.nrrd'
    image = sitk.ReadImage(segmentation_test_dir)
    
    truth = sitk.GetArrayFromImage(image)
    
    # plt.imshow(truth[0], cmap = 'gray')
    # plt.colorbar()
    # plt.show()
    
    truth_array.append(truth[0])
    

masks = []

directory = r"D:\Facultate\Licenta\Baza de imagini\ARCHIVE-RO01-04-20190813\dicoms\boala"
source_imgs = utils.load_scan(directory)

imgs = source_imgs
imgs = [ utils.crop_image(img, 205, 189, 200, 172)for img in imgs]


segmentations = [ utils.manual_filter(img, 175, 256).astype(np.uint8) for img in imgs]

for segmentation in segmentations :
    
    masca_dilatata = dilation(segmentation, disk(5))    
    # masca_finala = binary_fill_holes(masca_inchisa).astype(np.uint8)
    masca_finala = erosion(masca_dilatata, disk(3)).astype(np.uint8)
    
    masks.append(masca_finala)

gauri_gasite = []

index_imagine = 0

for index_imagine in range(len(imgs)):
    # print('---------------IMAGINE '+str(index_imagine)+'------------------')
    # print()
    img = imgs[index_imagine]
    mask = segmentations[index_imagine] 
    
    plt.imshow(img, cmap = 'gray')
    plt.colorbar()
    plt.show()
    plt.imshow(mask, cmap = 'gray')
    plt.colorbar()
    plt.show()
    labeled_mask_region, nr_label = label(mask,  return_num='True')

    gaura_gasita = np.zeros_like(img)
    # print("ajunge aici")
    for i in range(1,nr_label): # iterare prin fiecare zona de os din fiecare imagine 
        
        # print(f'regiunea {i}')
        # print()
    
        masca_regiune_curenta_os = (labeled_mask_region == i).astype(np.uint8)
        
        regiune_curenta_imagine_ct = img * masca_regiune_curenta_os
        
        labeled_masca_regiune_curenta_os = regionprops(masca_regiune_curenta_os, regiune_curenta_imagine_ct)
        aria_os_regiune_curenta = labeled_masca_regiune_curenta_os[0].area
        intensitate_medie_os_regiune_curenta = labeled_masca_regiune_curenta_os[0].mean_intensity
        
        blurred = filters.gaussian(regiune_curenta_imagine_ct, sigma=5) 
        img_regiune_curenta_margini_pronuntate = regiune_curenta_imagine_ct + (regiune_curenta_imagine_ct - blurred).astype(np.uint8)
        
        fig, axs = plt.subplots(1, 3, figsize=(12,12))
        
        axs[0].imshow(img, cmap='gray', vmin = 0, vmax = 255)
        axs[0].set_title(f'Imaginea {index_imagine}')
        
        axs[1].imshow(regiune_curenta_imagine_ct, cmap='gray', vmin = 0, vmax = 255)
        axs[1].set_title(f'Regiunea {i}/{nr_label-1}')
        
        axs[2].imshow(mask, cmap='gray')
        axs[2].set_title('Masca imaginii initiale')
        
        plt.show()
    
        laplacian = filters.laplace(img_regiune_curenta_margini_pronuntate)
        laplacian = filters.median(laplacian, selem=square(3))
        
        masca_regiune_curenta_os_erodata = erosion(masca_regiune_curenta_os, square(5))
        laplacian_erodat = np.where(masca_regiune_curenta_os_erodata == 1, laplacian, 0)   
        
        lapacian_echilibrat = laplacian_erodat
        
        val_prag = np.percentile(lapacian_echilibrat, 1)
        # print('Valoarea pragului: ' + str(val_prag))
        
        # plt.hist(laplacian.ravel(), bins='auto', color='blue', alpha=0.7)
        # plt.title('Histograma contrastului local')
        # plt.xlabel('Valoare')
        # plt.ylabel('Numărul de pixeli')
        # plt.show()

        # plt.hist(equlized_laplacian.ravel(), bins='auto', color='blue')
        # plt.title('Histograma equlized_laplacian')
        # plt.xlabel('Valoare')
        # plt.ylabel('Numărul de pixeli')
        # plt.show()   
        
        # plt.hist(laplacian.ravel(), bins='auto', color='blue')
        # plt.title('Histograma laplacian')
        # plt.xlabel('Valoare')
        # plt.ylabel('Numărul de pixeli')
        # plt.show() 
        
        gauri_laplacian = (lapacian_echilibrat < val_prag).astype(np.uint8)
        gauri_laplacian_labeled, nr_gauri = label(gauri_laplacian, return_num = True)
        
        min_laplacian = np.min(laplacian)
        max_laplacian =  np.max(laplacian)
        
        fig, axs = plt.subplots(1, 3, figsize=(12,12))
                
        axs[0].imshow(laplacian, cmap='gray', vmin = min_laplacian, vmax = max_laplacian )
        axs[0].set_title('laplacian img' + str(index_imagine))
        
        axs[1].imshow(lapacian_echilibrat, cmap='gray', vmin = min_laplacian, vmax = max_laplacian )
        axs[1].set_title('equlized_laplacian img'  + str(index_imagine))
        
        axs[2].imshow(gauri_laplacian_labeled, cmap='nipy_spectral')
        axs[2].set_title('Laplacian binarizat img'  + str(index_imagine))
        
        plt.show()
        
        fig, axs = plt.subplots(1, 2)
        
        axs[0].imshow(gauri_laplacian, cmap = 'gray')
        axs[0].set_title('Zone posibile')
        axs[1].imshow(img, cmap = 'gray')
        axs[1].set_title('Zona osoasa curenta')
        plt.show()
        
        imaginile_flood_fill_cu_gaura = []
        lista_masti_gauri_laplacian_fill = []
        
        # print('GAURILE')
        for j in range(1, nr_gauri):

            masca_gaura_laplacian = np.where(gauri_laplacian_labeled == j, 1, 0)
            gaura_laplacian_regions = regionprops(masca_gaura_laplacian)
            
            coordonate_gaura_laplacian = gaura_laplacian_regions[0].coords
            intensitati_gauri_laplacian = regiune_curenta_imagine_ct[coordonate_gaura_laplacian[:, 0], coordonate_gaura_laplacian[:, 1]]
            min_intensity_indices = np.where(intensitati_gauri_laplacian == np.min(intensitati_gauri_laplacian))[0]
            centroid_gaura_laplacian = np.array(gaura_laplacian_regions[0].centroid)
            distante_centroid_pcte_minime = np.linalg.norm(coordonate_gaura_laplacian[min_intensity_indices] - centroid_gaura_laplacian, axis=1)
            index_punct_apropiat = min_intensity_indices[np.argmin(distante_centroid_pcte_minime)]
            punct_start_centru_gaura_laplacian = tuple(coordonate_gaura_laplacian[index_punct_apropiat])
            
            
            imagine_flood_fill_cu_gaura = img_regiune_curenta_margini_pronuntate.copy()
            imagine_flood_fill_cu_gaura[img_regiune_curenta_margini_pronuntate == 0] = 1
            imagine_flood_fill_cu_gaura = flood_fill(imagine_flood_fill_cu_gaura, punct_start_centru_gaura_laplacian, new_value=0, tolerance=15, connectivity=1)
            
            imaginile_flood_fill_cu_gaura.append(imagine_flood_fill_cu_gaura)

            masca_gaura_flood_fill = (imagine_flood_fill_cu_gaura == 0).astype(np.uint8)


            
            masca_gaura_laplacian_fill = np.logical_or(masca_gaura_laplacian > 0, masca_gaura_flood_fill > 0).astype(np.uint8)
            
            masca_gaura_laplacian_fill_dilatata = dilation(masca_gaura_laplacian_fill, disk(1))
            masca_gaura_laplacian_fill = erosion(masca_gaura_laplacian_fill_dilatata, disk(1))
            lista_masti_gauri_laplacian_fill.append(masca_gaura_laplacian_fill)

            # fig, axs = plt.subplots(1, 4, figsize=(14,14))
            
            # axs[0].imshow(masca_gaura_laplacian, cmap='nipy_spectral')
            # axs[0].set_title(f'Puncte threshold gaura {j} img {index_imagine}')
            
            # axs[1].imshow(regiune_interes, cmap='gray', vmin = 0, vmax = 255)
            # axs[1].scatter(point[1], point[0], c='red', alpha=0.5)
            # axs[1].set_title(f'Pct {j} thresh. pe zona de interes img {index}')
            
            # axs[2].imshow(hole, cmap='gray')
            # axs[2].scatter(point[1], point[0], c='red', alpha=0.5, vmin = 0, vmax = 255)
            # axs[2].set_title(f'Gaura {j} in img {index}')
            
            # # Second subplot for 'closed_mask'
            # axs[3].imshow(closed_mask_hole, cmap='gray')
            # axs[3].set_title(f'Masca gaurii {j} img {index}')
            
            # plt.show() 
            
            props = regionprops(masca_gaura_laplacian_fill, intensity_image=regiune_curenta_imagine_ct)
            # conditii daca e tesut bolnav sau nu
            
            
            aria_gaura = props[0].area
            excentricitate_gaura = props[0].eccentricity
            soliditate_gaura = props[0].solidity
            intensitate_medie_gaura = props[0].mean_intensity
            
            intensity_values = props[0].intensity_image.flatten()
            std_dev = np.std(intensity_values)
            
            min_row, min_col, max_row, max_col = props[0].bbox
            axa1 = max_row - min_row
            axa2 = max_col - min_col
            lungime = np.max([axa1,axa2])
            latime = np.min([axa1,axa2])
            raport = latime/lungime
            
            if aria_gaura>4 \
                and intensitate_medie_gaura < intensitate_medie_os_regiune_curenta \
                and aria_gaura < 0.015 * aria_os_regiune_curenta \
                and raport < 1.5 and raport > 0.3  \
                and soliditate_gaura >0.80 :
                # and excentricitate_gaura < 0.8 \
                #  
            # and std_dev<100 \
            # 

            
    
                # print('asta e alta GAURA BUNA')
                
                # plt.imshow(masca_gaura_laplacian_fill, cmap = 'gray')
                # plt.colorbar()
                # plt.show()
                
                # print(f'intensitate_medie_os {intensitate_medie_os_regiune_curenta}')
                # print(f'intensitate_medie_gaura {intensitate_medie_gaura}')
                # print(f'aria_gaura{aria_gaura}')
                # print(f'excentricitate_gaura{excentricitate_gaura}')
                # print(f'soliditate_gaura{soliditate_gaura}')
                # print(f'std_dev{std_dev}')
                # print(f'raport{latime/lungime}')
                
                gaura_gasita = np.logical_or(gaura_gasita, masca_gaura_laplacian_fill).astype(np.uint8)               
                # fig, axs = plt.subplots(1, 4, figsize=(14,14))
                
                # axs[0].imshow(single_hole_image, cmap='nipy_spectral')
                # axs[0].set_title(f'Puncte threshold gaura {j} img {index_imagine}')
                
                # axs[1].imshow(regiune_curenta_imagine_ct, cmap='gray', vmin = 0, vmax = 255)
                # axs[1].scatter(point[1], point[0], c='red', alpha=0.5)
                # axs[1].set_title(f'Pct {j} thresh. pe zona de interes img {index_imagine}')
                
                # axs[2].imshow(hole, cmap='gray')
                # axs[2].scatter(point[1], point[0], c='red', alpha=0.5, vmin = 0, vmax = 255)
                # axs[2].set_title(f'Gaura {j} in img {index_imagine}')
                
                # axs[3].imshow(closed_mask_hole, cmap='gray')
                
                # axs[3].set_title(f'Masca gaurii {j} img {index_imagine}')
                
                # plt.show() 
       
            # else:
            #     print('NU E BUNA NNU E BUNA')
                
                # print(f'intensitate_medie_os {intensitate_medie_os_regiune_curenta}')
                # print(f'intensitate_medie_gaura {intensitate_medie_gaura}')
                # print(f'aria_gaura{aria_gaura}')
                # print(f'excentricitate_gaura{excentricitate_gaura}')
                # print(f'soliditate_gaura{soliditate_gaura}')
                # print(f'std_dev{std_dev}')
                            
                # fig, axs = plt.subplots(1, 4, figsize=(14,14))
                
                # axs[0].imshow(single_hole_image, cmap='nipy_spectral')
                # axs[0].set_title(f'Puncte threshold gaura {j} img {index_imagine}')
                
                # axs[1].imshow(regiune_curenta_imagine_ct, cmap='gray', vmin = 0, vmax = 255)
                # axs[1].scatter(point[1], point[0], c='red', alpha=0.5)
                # axs[1].set_title(f'Pct {j} thresh. pe zona de interes img {index_imagine}')
                
                # axs[2].imshow(hole, cmap='gray')
                # axs[2].scatter(point[1], point[0], c='red', alpha=0.5, vmin = 0, vmax = 255)
                # axs[2].set_title(f'Gaura {j} in img {index_imagine}')
                
                # axs[3].imshow(closed_mask_hole, cmap='gray')
                # axs[3].set_title(f'Masca gaurii {j} img {index_imagine}')
                
                # plt.show()    
    gauri_gasite.append(gaura_gasita)
    
masti_reconstruite = []

original_shape = source_imgs[0].shape
for gaura_gasita in gauri_gasite:
    masca_reconstruita = utils.insert_subimage_into_original(original_shape, gaura_gasita,  186, 172, 192, 189)
    masti_reconstruite.append(masca_reconstruita)
   

imagini_contur = []  
imagini_cerc = []              
for img, mask in zip(source_imgs, masti_reconstruite):
    
    img = np.stack([img] * 3, axis=2)
    
    contururi = measure.find_contours(mask, level=0.8)

    imagine_contur = img.copy()
    imagine_cerc = img.copy()

    for contur in contururi:
        rr, cc = draw.polygon_perimeter(contur[:, 0].astype(int), contur[:, 1].astype(int))
        
        imagine_contur[rr, cc, 0] = 255  # Canalul rosu
        imagine_contur[rr, cc, 1] = 0    # Canalul verde
        imagine_contur[rr, cc, 2] = 0    # Canalul albastru
        
        cx, cy = np.mean(contur, axis=0)
        
        raza = np.max(np.sqrt((contur[:, 0] - cx)**2 + (contur[:, 1] - cy)**2))
        
        rr, cc = draw.circle_perimeter(int(cx), int(cy), int(raza))
        imagine_cerc[rr, cc, 0] = 255  # Canalul rosu
        imagine_cerc[rr, cc, 1] = 0    # Canalul verde
        imagine_cerc[rr, cc, 2] = 255  # Canalul albastru
    
    imagini_contur.append(imagine_contur)
    imagini_cerc.append(imagine_cerc)


acuratete_segmentare = []
f1_segmentare = []
tn_segmentare = []
fp_segmentare = []
fn_segmentare = []
tp_segmentare = []
precizia_segmentare = []
recall_segmentare = [] 
nr_pixeli = imgs[0].shape[0]*imgs[0].shape[1]  
 
for i in range(len(imgs)):
    print(i)

    acuratete, precizie, recall, f1, TN, FP, FN, TP = compute_metrics(truth_array[i].flatten(), masti_reconstruite[i].flatten())
        
    acuratete_segmentare.append(acuratete)
    f1_segmentare.append(f1)
    recall_segmentare.append(3*recall)
    precizia_segmentare.append(3*precizie)
    tn_segmentare.append(TN/nr_pixeli)
    fp_segmentare.append(FP/nr_pixeli/3)
    fn_segmentare.append(FN/nr_pixeli)
    tp_segmentare.append(3*TP/nr_pixeli)

num_images = len(imgs)      

plt.figure(figsize=(10, 6))
plt.plot(range(num_images), fp_segmentare, marker='x', color='r', label='Fals pozitiv (FP)')
plt.plot(range(num_images), tp_segmentare, marker='d', color='m', label='Adevarat pozitiv (AP)')
plt.title('Fals și adevărat pozitiv pentru metoda segmentării leziunilor')
plt.xlabel('Numărul imaginii')
plt.ylabel('Rata de clasificare')
plt.legend()
plt.show()

plt.show()
text = [f'Img{i}' for i in range(num_images)]

media_acuratete = np.mean(acuratete_segmentare)
media_precizie = np.mean(precizia_segmentare)
media_rapel = np.mean(recall_segmentare)
plt.figure(figsize=(10, 6))
plt.barh(text, recall_segmentare, color='green')
plt.axvline(x=media_rapel, color='r', linestyle='--', label=f'Medie: {media_rapel:.2f}')
plt.xlabel('Rapel')
plt.ylabel('Poze')
plt.title('Rapelul segmentarii leziunilor pentru fiecare poza')
plt.yticks([0, len(text)//2, len(text)-1], [text[0], text[len(text)//2], text[-1]])
plt.legend()
plt.show()





