import pydicom as dicom
import os
import numpy as np

def load_scan(path):
    
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                data.append(dicom.read_file(dicom_path))
    
    data.sort(key = lambda x: int(x.InstanceNumber))           
    
    biti = 8
    datatype = np.uint8   
    
    (image_height, image_width)= (0,0)
    (image_height, image_width) = np.shape(data[0].pixel_array)
    n = len(data)
    
    intensitate_maxima = 2**biti - 1 
    
    images = np.zeros((n, image_height, image_width), dtype = datatype)
    
    for i , d in enumerate(data):

        try:
            image =  d.pixel_array
            
            min_val = np.min(image)
            max_val = np.max(image)
            
            # plt.imshow(image, cmap = 'gray', vmin = np.min(image), vmax=np.max(image) )
            # plt.colorbar()
            # plt.title('img la inceput')
            # plt.show()
    
            image = np.clip((image - min_val) / (max_val - min_val) * intensitate_maxima, 0 ,intensitate_maxima).astype(datatype)
                    
            # plt.imshow(image, cmap = 'gray', vmin = np.min(image), vmax=np.max(image) )
            # plt.colorbar()
            # plt.title('img dupa ce s-a facut shiftarea')
            # plt.show()
    
            images[i] = image
         
        except ValueError: 
            pass 
                

    
    return images

# with open('variabile.pickle', 'wb') as f:
#     pickle.dump((a, b, c), f)

# with open('variabile.pickle', 'rb') as f:
#     a, b, c = pickle.load(f)

def manual_filter(img, left, right):

    return np.where((img < left) | (img > right), 0, 2**8 -1)


def crop_image(image, left, right, top, bottom):


    height, width = image.shape

    # verifica daca numarul de pixeli taiati nu depaseste dimensiunea imaginii
    left = min(left, width-1)
    right = min(right, width-1)
    top = min(top, height-1)
    bottom = min(bottom, height-1)

    return image[top:-bottom if bottom != 0 else None, left:-right if right != 0 else None]

def insert_subimage_into_original(original_shape, subimage, top, bottom, left, right):

    new_image = np.zeros(original_shape, dtype=np.uint8)

    new_image[top:original_shape[0] - bottom, left:original_shape[1] - right] = subimage
    
    return new_image

def normalize_img(imgs):


    for i in range(len(imgs)):
        
        img = imgs[i]
        
        min_intensity = np.min(img)
        
        # normalizarea intensității pixelilor prin scalare
        imgs[i] = img - min_intensity

    return imgs

def get_paths(director):

    files = os.listdir(director)

    directoare1 = []
    for file in files:
        directoare1.append(os.path.join(director, file))
    toate = []
    for director1 in directoare1:
        
        files1 = os.listdir(director1)
        
        for file1 in files1:
            toate.append(os.path.join(director1, file1))
    
    return toate



