import os
import utils 
import matplotlib.pylab as plt
import numpy as np 
from sklearn.cluster import KMeans

def dicom_paths(cale):

    out = []    

    if not os.path.exists(cale):
        print("Calea specificată nu există!")
        return

    for root, dirs, files in os.walk(cale):
        for nume_fisier in files:

            cale_completa = os.path.join(root, nume_fisier)
            
            if cale_completa[-3:] == 'dcm':
                out.append(root)
                
    return list(set(out)) # sterg duplicate 

start_path = r"D:\Facultate\Licenta\Baza de imagini\manifest-1669911955156" # Modifică această cale cu calea ta

paths = dicom_paths(start_path)

all_imgs = []

cropped_imgs = [utils.crop_image(img, 28, 0, 70, 121) for img in utils.load_scan(paths[0]) ] 
all_imgs.extend(cropped_imgs)

cropped_imgs = utils.load_scan(paths[3])
all_imgs.append(cropped_imgs)

cropped_imgs = utils.load_scan(paths[4])
all_imgs.append(cropped_imgs)

cropped_imgs = utils.load_scan(paths[5])
all_imgs.append(cropped_imgs)

cropped_imgs = [utils.crop_image(img, 7, 0, 70, 70) for img in utils.load_scan(paths[10]) ]
all_imgs.extend(cropped_imgs)

cropped_imgs = [utils.crop_image(img, 31, 4, 0, 102) for img in utils.load_scan(paths[14]) ]
all_imgs.extend(cropped_imgs)

cropped_imgs = [utils.crop_image(img, 39, 0, 0, 87) for img in utils.load_scan(paths[15]) ]
all_imgs.extend(cropped_imgs)

cropped_imgs = [utils.crop_image(img, 0, 0, 0, 105) for img in utils.load_scan(paths[20]) ]
all_imgs.extend(cropped_imgs)

cropped_imgs = [utils.crop_image(img, 0, 0, 0, 70) for img in utils.load_scan(paths[24]) ]
all_imgs.extend(cropped_imgs)

cropped_imgs = utils.load_scan(paths[25])
all_imgs.append(cropped_imgs)

cropped_imgs = [utils.crop_image(img, 0, 0, 0, 70) for img in utils.load_scan(paths[24]) ]
all_imgs.extend(cropped_imgs)

cropped_imgs = [utils.crop_image(img, 115, 0, 0, 0) for img in utils.load_scan(paths[26]) ]
all_imgs.extend(cropped_imgs)

cropped_imgs = utils.load_scan(paths[27])
all_imgs.append(cropped_imgs)

cropped_imgs = utils.load_scan(paths[28])
all_imgs.append(cropped_imgs)

pixels = [img.reshape(-1) for img in all_imgs]

np.random.shuffle(pixels)

print('kmeans')

centroizi_init = np.random.rand(4, 1)

for count, array  in pixels:
    print(str(count) + '/' + str(len(pixels)))
    kmeans = KMeans(n_clusters=4, init=centroizi_init, n_init=1, random_state=18)
    kmeans.fit(array)

    centroizi_init = kmeans.cluster_centers_


##### RESULTS 0.415512, 151.183, 216.824 , 94.0981


# ####### PROBA KMEANS

# centers = []
# i = 0

# for img in imgs:
    
#     kmeans = KMeans(n_clusters=4, init='k-means++',  ).fit(np.reshape(img,[np.prod(img.shape),1]))
    
#     # Predictia centroizilor pentru fiecare grup
#     # print(f"Centroizii grupurilor pentru img {i}:")
#     # print(kmeans.cluster_centers_)
#     centers.append(sorted(kmeans.cluster_centers_, reverse= True))
#     # Predictia grupurilor pentru fiecare punct de date
#     # print(f"Predictiile grupurilor pentru img {i}sunt:")
#     plt.imshow(kmeans.labels_.reshape(512,512) )
#     plt.colorbar()
#     plt.title('middle')
#     plt.show()
#     i = i+1
# centers = sorted(centers, key=lambda centers: centers[0], reverse = True) #### sorteaza centroizii dupa primul element. 
        