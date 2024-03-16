import utils
from tkinter import Canvas, Label, StringVar
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from sklearn.cluster import KMeans
from skimage.morphology import erosion, square, dilation, disk, flood_fill
from skimage.measure import label, regionprops
from skimage import filters, measure
from skimage import draw
from tkinter import filedialog
import os

class NewLabel(Label):
    def __init__(self, master=None, parent = None, **kwargs):
        super().__init__(master, **kwargs)
       
        self.parent = parent
        self.counter = 0
        self.max_count = 0  
        self.label_var = StringVar()
        self.label_var.set(f"{self.counter + 1} / {self.max_count}") 
        self.config(textvariable=self.label_var)

    def set_max_count(self, max_count):
        self.max_count = max_count
        self.label_var.set(f"{self.counter + 1} / {self.max_count}")

    def increase(self):
        if self.counter < self.max_count - 1:
            self.counter += 1
        else:
            self.counter = 0
        self.label_var.set(f"{self.counter + 1} / {self.max_count}")
        self.parent.update_image()

    def decrease(self):
        if self.counter > 0:
            self.counter -= 1
        else:
            self.counter = self.max_count - 1
        self.label_var.set(f"{self.counter + 1} / {self.max_count}")
        self.parent.update_image()
    
class ImageCanvas(Canvas):
    def __init__(self, master=None,  **kwargs):
        super().__init__(master, **kwargs)
        
    def display_image(self, img_to_show):
        
        if len(img_to_show.shape) == 2:
            img_to_show = (img_to_show / img_to_show.max() * 255).astype(np.uint8)
            img = Image.fromarray(img_to_show)
            img = img.convert("L")
            
        if len(img_to_show.shape) == 3:
            img = Image.fromarray(img_to_show.astype('uint8'), 'RGB') 
        
        img_width, img_height = img.size
        img = img.resize((img_width // 2, img_height // 2), Image.Resampling.LANCZOS)

        # Resize canvas to fit the image
        self.config(width=img_width//2, height=img_height//2)
        photo = ImageTk.PhotoImage(img)
        self.create_image(0, 0, image=photo, anchor='nw')
        self.image = photo
        
class ImageFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master,  highlightbackground="gray", highlightthickness=5 , **kwargs)
        
        self.imgs = []
        
        self.canvas = ImageCanvas(self, width=250, height=250)
        self.label = NewLabel(self, parent = self)
        
        self.buffer = tk.Frame(self, height=50)
        self.buffer.pack_propagate(False)  # împiedică redimensionarea automată
        self.label_max_count = tk.Label(self, text="") 

        self.button_previous = tk.Button(self, text="<-", command=self.label.decrease)
        self.button_next = tk.Button(self, text="->", command=self.label.increase)
        
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.button_previous.grid(row=1, column=0, sticky = 'w')
        self.label.grid(row=1, column=1, sticky='we')
        self.button_next.grid(row=1, column=2, sticky = 'e')
        self.buffer.grid(row = 2, columnspan = 3 )
        
        self.grid_columnconfigure(1, weight=1) # va da coloanei cu label (1) mai mult spațiu
        self.grid_columnconfigure(0, weight=0) # va impiedica butonul stanga sa se extinda
        self.grid_columnconfigure(2, weight=0) # va impiedica butonul dreapta sa se extinda
                
    def load_images(self, images):
        self.imgs = images
        self.label.set_max_count(len(self.imgs))
        self.update_image()
        
        # modifica dimensiunea frame in functie de canvas si butoane
        canvas_width = self.canvas.winfo_reqwidth()  
        canvas_height = self.canvas.winfo_reqheight() 
            
        self.button_previous.update_idletasks() 
        button_height = self.button_previous.winfo_height()
    
        # dimensiuni frame
        frame_width = canvas_width  
        frame_height = canvas_height + button_height 
    
        self.config(width=frame_width, height=frame_height)

        self.update_idletasks()
        
    def update_image(self):
        current_img = self.imgs[self.label.counter]
        print(f"Current image shape: {current_img.shape}")
        self.canvas.display_image(current_img)

class CropFrame(ImageFrame):
    def __init__(self, master=None, length=400, images=None, update_callback = None, **kwargs):
        super().__init__(master,  **kwargs)
        
        self.update_callback = update_callback
        self.length = length
        self.images = images

        self.right_scale = None
        self.left_scale = None
        self.top_scale = None
        self.bottom_scale = None
        
        self.crop_button = tk.Button(self.buffer, text = "Crop Image", command = self.crop_images)
        self.reset_button  = tk.Button(self.buffer, text = "Reset Image", command = self.reset_images)
        self.crop_button.grid(row=4, column=0, sticky='ew')
        self.reset_button .grid(row=4, column=1, sticky='ew')

    def on_scale_change(self, val):
        right = self.right_scale.get()
        left = self.left_scale.get()
        up = self.top_scale.get()
        down = self.bottom_scale.get()
        counter = self.label.counter
        
        current_img = self.images[counter]
        
        cropped_img = utils.crop_image(current_img, left, right, up, down)
        self.canvas.display_image(cropped_img)
        
        
    def update_image(self):
        current_img = self.images[self.label.counter]
        right = self.right_scale.get()
        left = self.left_scale.get()
        up = self.top_scale.get()
        down = self.bottom_scale.get()
        
        cropped_img = utils.crop_image(current_img, left, right, up, down)
        self.canvas.display_image(cropped_img)

    def load_images(self, images):
            self.images = images
            self.back_up_imgs = images
            self.now_img = images[0]
            self.label.set_max_count(len(self.images))
    
            canvas_width = self.canvas.winfo_reqwidth()
            canvas_height = self.canvas.winfo_reqheight()
    
            self.button_previous.update_idletasks()
            button_height = self.button_previous.winfo_height()
    
            frame_width = canvas_width
            frame_height = canvas_height + button_height
    
            self.config(width=frame_width, height=frame_height)
            self.update_idletasks()
    
            self.buffer.config(width=self.winfo_width())
    
            if self.right_scale:
                self.right_scale.destroy()
    
            if self.left_scale:
                self.left_scale.destroy()
                
            if self.top_scale:
                self.top_scale.destroy()
    
            if self.bottom_scale:
                self.bottom_scale.destroy()
            
            max_hor , max_vert = self.now_img.shape
            max_hor = max_hor // 2 
            max_vert = max_vert // 2

            self.right_scale = tk.Scale(self.buffer, from_=0, label = 'Right', to=max_vert, orient="horizontal", length=200, command=self.on_scale_change)
            self.left_scale = tk.Scale(self.buffer, from_=0, label = 'Left', to=max_vert-1, orient="horizontal", length=200, command=self.on_scale_change)
            self.top_scale = tk.Scale(self.buffer, from_=0, label = 'Up', to=max_hor-1, orient="horizontal", length=200, command=self.on_scale_change)
            self.bottom_scale = tk.Scale(self.buffer, from_=0, label = 'Down', to=max_hor, orient="horizontal", length=200, command=self.on_scale_change)

            self.right_scale.grid(row=0, column=0, columnspan=2, sticky='ew')
            self.left_scale.grid(row=1, column=0, columnspan=2, sticky='ew')
            self.top_scale.grid(row=2, column=0, columnspan=2, sticky='ew')
            self.bottom_scale.grid(row=3, column=0, columnspan=2, sticky='ew')

            self.update_idletasks()
    
            self.update_image()
        
    def crop_images(self):
        right = self.right_scale.get()
        left = self.left_scale.get()
        up = self.top_scale.get()
        down = self.bottom_scale.get()

        cropped_img = [utils.crop_image(img, left, right, up, down) for img in self.images]
        self.images = cropped_img
        
        if self.update_callback is not None:
            self.update_callback(self.images, self.right_scale.get(), self.left_scale.get(), self.top_scale.get(), self.bottom_scale.get(),action="cut", )
        
        max_hor , max_vert = self.images[0].shape
        max_hor = max_hor // 2 
        max_vert = max_vert // 2   
        
        self.right_scale.set(0)
        self.left_scale.set(0)
        self.top_scale.set(0)
        self.bottom_scale.set(0)
        self.right_scale.configure(to=max_vert)
        self.left_scale.configure(to=max_vert-1)
        self.top_scale.configure(to=max_hor-1)
        self.bottom_scale.configure(to=max_hor)

    def reset_images(self):
        
        counter = self.label.counter
        self.canvas.display_image(self.back_up_imgs[counter])
        self.images = self.back_up_imgs
        
        if self.update_callback is not None:
            self.update_callback(self.images, 0, 0, 0, 0, action="revert", )  # Scale-uri resetate la 0
        
        max_hor , max_vert = self.images[0].shape
        max_hor = max_hor // 2 
        max_vert = max_vert // 2   
        
        self.right_scale.set(0)
        self.left_scale.set(0)
        self.top_scale.set(0)
        self.bottom_scale.set(0)
        self.right_scale.configure(to=max_vert)
        self.left_scale.configure(to=max_vert)
        self.top_scale.configure(to=max_hor)
        self.bottom_scale.configure(to=max_hor)
               
class CreateMaskFrame(ImageFrame):
    def __init__(self, master=None, length=400, images=None, update_callback = None, **kwargs):
        super().__init__(master, **kwargs)
        self.length = length
        self.images = images
        
        self.update_callback = update_callback
        
        self.mask_button = tk.Button(self.buffer, text="Apply Mask", command=self.create_mask)
        self.mask_button.grid(row = 4, column = 0, columnspan=2,sticky='n')
        
        self.k_means_check_var = tk.IntVar()
        self.manual_check_var = tk.IntVar(value = 1)
        self.mac_check_var = tk.IntVar()
        
        self.manual_scales_state = True

    def on_scale_change(self, val):
        left = self.scale1.get()
        right = self.scale2.get()
        counter = self.label.counter
        
        current_img = self.images[counter]
    
        self.canvas.display_image(utils.manual_filter(current_img, left, right))
        
    def update_image(self):
        current_img = self.images[self.label.counter]
        left = self.scale1.get()
        right = self.scale2.get()
        filtered_img = utils.manual_filter(current_img, left, right)
        self.canvas.display_image(filtered_img)

    def load_images(self, images):
            self.images = images
            self.now_img = [0]
            self.label.set_max_count(len(self.images))
    
            canvas_width = self.canvas.winfo_reqwidth()
            canvas_height = self.canvas.winfo_reqheight()
    
            self.button_previous.update_idletasks()
            button_height = self.button_previous.winfo_height()
    
            frame_width = canvas_width
            frame_height = canvas_height + button_height
    
            self.config(width=frame_width, height=frame_height)
            self.update_idletasks()
    
            self.buffer.config(width=self.winfo_width())
            
            self.scale1 = tk.Scale(self.buffer, from_=0, to=2**8, orient="horizontal", length=200, command=self.on_scale_change)
            self.scale2 = tk.Scale(self.buffer, from_=2, to=2**8, orient="horizontal", length=200, command=self.on_scale_change)

            self.scale1.set(177)      # seteaza valoarea inițială pt o evidentiere mai buna a functionalitatii
            self.scale2.set(256)  
        
            self.k_means_box = tk.Checkbutton(self.buffer, text="K-Means", variable=self.k_means_check_var)
            self.manual_box = tk.Checkbutton(self.buffer, text="Manual", variable=self.manual_check_var, command = self.toggle_manual_scale)
            self.mac_box = tk.Checkbutton(self.buffer, text="Morphological active contour", variable=self.mac_check_var)
        
            self.scale1.grid(row=0, column=0, sticky='ew', columnspan = 2)
            self.scale2.grid(row=1, column=0, sticky='ew', columnspan = 2)
            self.k_means_box.grid(row=2, column=0, sticky='w')
            self.manual_box.grid(row=2, column=1, sticky='e')
            self.mac_box.grid(row=3, column=0, sticky='e')
    
            self.update_idletasks()
    
            self.update_image()
            
    def toggle_manual_scale(self):
        
        self.manual_scales_state = not self.manual_scales_state
        
        if self.manual_scales_state:
            self.scale1.grid()
            self.scale2.grid()
        else:
            self.scale1.grid_remove()
            self.scale2.grid_remove()

    def create_mask(self ):
        check1 = self.k_means_check_var.get()
        check2 = self.manual_check_var.get()
        check3 = self.mac_check_var.get()
        
        original_images = self.images
        
        if (check1 == 0 and check2 == 1 and check3 == 0):  #  MANUAL SEGMENTATION
            if self.update_callback is not None:
                
                masked_images = [] 
                left = self.scale1.get()
                right = self.scale2.get()
                
            for img in self.images: 
                
                segmentare = utils.manual_filter(img, left, right ).astype(np.uint8)
                
                masca_dilatata = dilation(segmentare, disk(5))   
                
                masca_finala = erosion(masca_dilatata, disk(3)).astype(np.uint8)
                
                masked_images.append(masca_finala)
            
            self.update_callback(masked_imgs=masked_images, original_imgs=original_images)
            
        elif (check1 == 1 and check2 == 0 and check3 == 0):  #  KMEANS SEGMENTATION
            if self.update_callback is not None:
                
                masked_images = []
                centroizi_init = [0.415512, 151.183, 216.824 , 94.0981]
                centroizi_init = np.array(centroizi_init).reshape(-1, 1)

                for img in self.images:                
                    
                    pixels = img.reshape(-1, 1)
                    kmeans = KMeans(n_clusters=len(centroizi_init), init=np.array(centroizi_init), n_init=1, random_state=42)
                    etichete = kmeans.fit_predict(pixels)
                    
                    segmentation = etichete.reshape(img.shape)
                    
                    masca_osos = (segmentation == 2).astype(np.uint8)
                    
                    masca_dilatata = dilation(masca_osos, disk(5))   
                    
                    masca_finala = erosion(masca_dilatata, disk(3)).astype(np.uint8)
                    
                    masked_images.append(masca_finala)
                    
                self.update_callback(masked_imgs=masked_images, original_imgs=original_images)
                
        elif (check1 == 0 and check2 == 0 and check3 == 1):  # MORPHOLOGIC SEGMENTATION
            if self.update_callback is not None:
                
                masked_images = []
                
                for img in self.images:
                    
                    init_ls  = utils.manual_filter(img,self.scale1.get(), self.scale2.get())  
                    gimage = inverse_gaussian_gradient(img)
                    ls = morphological_geodesic_active_contour(gimage, 230, init_ls,
                                        smoothing=2, balloon=0,
                                        threshold=0.1)
                    
                    masked_images.append(ls)
                
                self.update_callback(masked_imgs=masked_images, original_imgs=original_images)
                
        else:
            messagebox.showerror("Eroare", "Alegeți un tip de mascare")      
            
class SegmentationFrame(ImageFrame):
    def __init__(self, master=None, update_callback = None, **kwargs):
        super().__init__(master,  **kwargs)
        
        self.images = []
        self.masked_images = []
        self.original_images = []
        self.update_callback = update_callback
        
        self.scale1_value = 0
        self.scale2_value = 0
        self.scale3_value = 0
        self.scale4_value = 0
        
        self.segmentation_button = tk.Button(self.buffer, text="Segment", command=self.segmentation)
        self.segmentation_button.grid(row = 0, column = 0, columnspan=2,sticky='n')
        
    def update_image(self):
        current_mask = self.masked_images[self.label.counter]
        self.canvas.display_image(current_mask)
        
    def load_images(self, masked_images, original_images):
        self.masked_images = masked_images
        self.images = original_images
        self.label.set_max_count(len(self.images))
        self.update_image()
        
        # modifica dimensiunea frame in functie de canvas si butoane
        canvas_width = self.canvas.winfo_reqwidth()  
        canvas_height = self.canvas.winfo_reqheight() 
            
        self.button_previous.update_idletasks() 
        button_height = self.button_previous.winfo_height()
    
        # dimensiuni frame
        frame_width = canvas_width  
        frame_height = canvas_height + button_height 
    
        self.config(width=frame_width, height=frame_height)

        self.update_idletasks()
        
    def update_scale_values(self, scale1_value, scale2_value, scale3_value, scale4_value, action):
        
        if action == 'cut':
            self.scale1_value = scale1_value + self.scale1_value
            self.scale2_value = scale2_value + self.scale2_value
            self.scale3_value = scale3_value + self.scale3_value
            self.scale4_value = scale4_value + self.scale4_value
            
        if action == 'revert':
            self.scale1_value = 0
            self.scale2_value = 0
            self.scale3_value = 0
            self.scale4_value = 0
        
    def segmentation(self):
        if self.update_callback is not None:

            found_holes = []
            
            image_index = 0
            
            for image_index in range(len(self.images)):
                # print('---------------IMAGE '+str(image_index)+'------------------')
                # print()
                image = self.images[image_index]
                mask = self.masked_images[image_index] 
                
                # plt.imshow(image, cmap = 'gray')
                # plt.colorbar()
                # plt.show()
                # plt.imshow(mask, cmap = 'gray')
                # plt.colorbar()
                # plt.show()
                labeled_mask_region, num_labels = label(mask,  return_num='True')
            
                found_hole = np.zeros_like(image)

                for i in range(1,num_labels): # iterating through each bone area in each image
                    
                    # print(f'REGION {i}')
                    # print()
                
                    current_bone_region_mask = (labeled_mask_region == i).astype(np.uint8)
                    
                    current_ct_image_region = image * current_bone_region_mask
                    
                    labeled_current_bone_region_mask = regionprops(current_bone_region_mask, current_ct_image_region)
                    current_bone_region_area = labeled_current_bone_region_mask[0].area
                    current_bone_region_mean_intensity = labeled_current_bone_region_mask[0].mean_intensity
                    
                    blurred = filters.gaussian(current_ct_image_region, sigma=5) 
                    current_region_edges_enhanced_image = current_ct_image_region + (current_ct_image_region - blurred).astype(np.uint8)
                    
                    # fig, axs = plt.subplots(1, 3, figsize=(12,12))
                    
                    # axs[0].imshow(image, cmap='gray', vmin = 0, vmax = 255)
                    # axs[0].set_title(f'Image {image_index}')
                    
                    # axs[1].imshow(current_ct_image_region, cmap='gray', vmin = 0, vmax = 255)
                    # axs[1].set_title(f'Region {i}/{num_labels-1}')
                    
                    # axs[2].imshow(mask, cmap='gray')
                    # axs[2].set_title('Initial image mask')
                    
                    # plt.show()
                
                    laplacian = filters.laplace(current_region_edges_enhanced_image)
                    laplacian = filters.median(laplacian, footprint=square(3))
                    
                    eroded_current_bone_region_mask = erosion(current_bone_region_mask, square(5))
                    eroded_laplacian = np.where(eroded_current_bone_region_mask == 1, laplacian, 0)   
                    
                    balanced_laplacian = eroded_laplacian
                    
                    threshold_value = np.percentile(balanced_laplacian, 1)
                    # print('Threshold value: ' + str(threshold_value))
                    
                    laplacian_holes = (balanced_laplacian < threshold_value).astype(np.uint8)
                    labeled_laplacian_holes, number_of_holes = label(laplacian_holes, return_num = True)
                    
                    # fig, axs = plt.subplots(1, 3, figsize=(12,12))
                            
                    # axs[0].imshow(laplacian, cmap='gray', vmin = min_laplacian, vmax = max_laplacian )
                    # axs[0].set_title('laplacian image' + str(image_index))
                    
                    # axs[1].imshow(balanced_laplacian, cmap='gray', vmin = min_laplacian, vmax = max_laplacian )
                    # axs[1].set_title('equlized_laplacian image'  + str(image_index))
                    
                    # axs[2].imshow(labeled_laplacian_holes, cmap='nipy_spectral')
                    # axs[2].set_title('Binarized Laplacian Image'  + str(image_index))
                    
                    # plt.show()
                    
                    # fig, axs = plt.subplots(1, 2)
                    
                    # axs[0].imshow(laplacian_holes, cmap = 'gray')
                    # axs[1].imshow(image, cmap = 'gray')
                    # plt.show()
                    
                    flood_filled_images_with_hole = []
                    list_of_laplacian_hole_masks_filled = []
                    
                    # print('HOLES')
                    for j in range(1, number_of_holes):

                        laplacian_hole_mask = np.where(labeled_laplacian_holes == j, 1, 0)
                        laplacian_hole_regions = regionprops(laplacian_hole_mask)
                        
                        laplacian_hole_coordinates = laplacian_hole_regions[0].coords
                        laplacian_hole_intensities = current_ct_image_region[laplacian_hole_coordinates[:, 0], laplacian_hole_coordinates[:, 1]]
                        min_intensity_indices = np.where(laplacian_hole_intensities == np.min(laplacian_hole_intensities))[0]
                        laplacian_hole_centroid = np.array(laplacian_hole_regions[0].centroid)
                        distances_from_centroid_to_min_points = np.linalg.norm(laplacian_hole_coordinates[min_intensity_indices] - laplacian_hole_centroid, axis=1)
                        nearest_point_index = min_intensity_indices[np.argmin(distances_from_centroid_to_min_points)]
                        start_point_center_of_laplacian_hole = tuple(laplacian_hole_coordinates[nearest_point_index])
                        
                        
                        flood_filled_image_with_hole = current_region_edges_enhanced_image.copy()
                        flood_filled_image_with_hole[current_region_edges_enhanced_image == 0] = 1
                        flood_filled_image_with_hole = flood_fill(flood_filled_image_with_hole, start_point_center_of_laplacian_hole, new_value=0, tolerance=15, connectivity=1)
                        
                        flood_filled_images_with_hole.append(flood_filled_image_with_hole)
            
                        flood_fill_hole_mask = (flood_filled_image_with_hole == 0).astype(np.uint8)
            

                        
                        filled_laplacian_hole_mask = np.logical_or(laplacian_hole_mask > 0, flood_fill_hole_mask > 0).astype(np.uint8)
                        
                        dilated_laplacian_hole_mask = dilation(filled_laplacian_hole_mask, disk(1))
                        filled_laplacian_hole_mask = erosion(dilated_laplacian_hole_mask, disk(1))
                        list_of_laplacian_hole_masks_filled.append(filled_laplacian_hole_mask)
                        
                        props = regionprops(filled_laplacian_hole_mask, intensity_image=current_ct_image_region)
                        # check for ill tissue
                        
                        hole_area = props[0].area
                        hole_solidity = props[0].solidity
                        hole_mean_intensity = props[0].mean_intensity
                        
                        min_row, min_col, max_row, max_col = props[0].bbox
                        axis1 = max_row - min_row
                        axis2 = max_col - min_col
                        length = np.max([axis1,axis2])
                        width = np.min([axis1,axis2])
                        aspect_ratio = width/length
                        
                        if hole_area>4 \
                            and hole_mean_intensity < current_bone_region_mean_intensity \
                            and hole_area < 0.015 * current_bone_region_area \
                            and aspect_ratio < 1.5 and aspect_ratio > 0.3  \
                            and hole_solidity >0.80 :
                            
                            found_hole = np.logical_or(found_hole, filled_laplacian_hole_mask).astype(np.uint8)               

                found_holes.append(found_hole)
                
            reconstructed_masks = []

            original_shape = self.original_images[0].shape
            for found_hole in found_holes:
                reconstructed_mask = utils.insert_subimage_into_original(original_shape, found_hole,  self.scale3_value, self.scale4_value, self.scale2_value, self.scale1_value)
                reconstructed_masks.append(reconstructed_mask)
   

            contour_images = []  
            circle_images = []              
            for image, mask in zip(self.original_images, reconstructed_masks):
                
                image = np.stack([image] * 3, axis=2)
                
                contours = measure.find_contours(mask, level=0.8)
            
                contour_image = image.copy()
                circle_image = image.copy()
            
                for contur in contours:
                    rr, cc = draw.polygon_perimeter(contur[:, 0].astype(int), contur[:, 1].astype(int))
                    
                    contour_image[rr, cc, 0] = 255 
                    contour_image[rr, cc, 1] = 0    
                    contour_image[rr, cc, 2] = 0    
                    
                    cx, cy = np.mean(contur, axis=0)
                    
                    raza = np.max(np.sqrt((contur[:, 0] - cx)**2 + (contur[:, 1] - cy)**2))
                    
                    rr, cc = draw.circle_perimeter(int(cx), int(cy), int(raza))
                    circle_image[rr, cc, 0] = 255 
                    circle_image[rr, cc, 1] = 0    
                    circle_image[rr, cc, 2] = 255  # Canalul albastru
                
                contour_images.append(contour_image)
                circle_images.append(circle_image)

    
        self.update_callback(circle_images, contour_images, reconstructed_masks)    
        
class ResultFrame(tk.Frame):

    def __init__(self, master=None, **kwargs):
        super().__init__(master,  highlightbackground="gray", highlightthickness=5 , **kwargs)
        
        self.contour_images = []
        self.circle_images = []
        self.reconstructed_masks = []
        
        self.canvas1 = ImageCanvas(self, width=250, height=250)
        self.canvas2 = ImageCanvas(self, width=250, height=250)
        self.label = NewLabel(self, parent = self)
        
        self.buffer = tk.Frame(self, height=50)  # setarea lățimii
        self.buffer.pack_propagate(False)  # împiedică redimensionarea automată
        self.label_max_count = tk.Label(self, text="") 
 
        self.button_previous = tk.Button(self, text="<-", command=self.label.decrease)
        self.button_next = tk.Button(self, text="->", command=self.label.increase)
        
        self.export_button = tk.Button(self, text="Export results", command=self.export_results)
        
        self.canvas1.grid(row=0, column=0, columnspan=2)
        self.canvas2.grid(row=0, column=2, columnspan=1)

        self.canvas1.grid(row=0, column=1)
        self.canvas2.grid(row=0, column=4)
        
        self.button_previous.grid(row=1, column=0, sticky = 'w')
        self.label.grid(row=1, column=3)
        self.button_next.grid(row=1, column=5,  sticky = 'e')

        self.export_button.grid(row=3, column=1, sticky = 'we')
                
    def load_images(self, contour_images, circle_images, reconstructed_masks):
        self.contour_images = contour_images
        self.circle_images = circle_images
        self.reconstructed_masks = reconstructed_masks
        self.label.set_max_count(len(self.contour_images))
        self.update_image()
        
        # modifica dimensiunea frame in functie de canvas si butoane
        canvas_width = self.canvas1.winfo_reqwidth()  
        canvas_height = self.canvas1.winfo_reqheight() 
            
        self.button_previous.update_idletasks() 
        button_height = self.button_previous.winfo_height()
    
        # dimensiuni frame
        frame_width = 2*canvas_width  
        frame_height = 2*(canvas_height + button_height) 
    
        self.config(width=frame_width, height=frame_height)

        self.update_idletasks()
        
    def update_image(self):
        current_img_contur = self.contour_images[self.label.counter]
        current_img_cerc = self.circle_images[self.label.counter]
        self.canvas1.display_image(current_img_contur)
        self.canvas2.display_image(current_img_cerc)
    
    def export_results(self):
        folder_selected = filedialog.askdirectory()
        for i in range(len(self.contour_images)):
            
            results_folder = os.path.join(folder_selected, 'results')
            os.makedirs(results_folder, exist_ok=True)
            
            masks_folder = os.path.join(results_folder, 'masks')
            os.makedirs(masks_folder, exist_ok=True)
            image_path = os.path.join(masks_folder, f"mask_{i+1}.jpg")
            img_to_save = Image.fromarray((self.reconstructed_masks[i] * 255).astype('uint8'), 'L')
            img_to_save.save(image_path)
            
            contours_folder = os.path.join(results_folder, 'contours')
            os.makedirs(contours_folder, exist_ok=True)
            image_path = os.path.join(contours_folder, f"contour_{i+1}.jpg")
            img_to_save = Image.fromarray((self.contour_images[i]).astype('uint8'), 'RGB')
            img_to_save.save(image_path)
            
            circled_folder = os.path.join(results_folder, 'circled')
            os.makedirs(circled_folder, exist_ok=True)
            image_path = os.path.join(circled_folder, f"circled_{i+1}.jpg")
            img_to_save = Image.fromarray((self.circle_images[i]).astype('uint8'), 'RGB')
            img_to_save.save(image_path)
            