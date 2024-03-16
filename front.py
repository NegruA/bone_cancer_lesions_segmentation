import tkinter as tk
from tkinter import filedialog
from back import CropFrame
from back import CreateMaskFrame 
from back import SegmentationFrame
from back import ResultFrame
import utils
import matplotlib.pylab as plt

class MyApplication:
    def __init__(self):
        
        self.initial_images  = []
        self.images_count = 0
        
        # Creating the main window
        self.window = tk.Tk()
        # Setting minimum and maximum window sizes
        self.window.minsize(1500, 600)
        self.window.maxsize(1500, 600)

        # Creating frame1 for image preview
        self.frame1 = tk.Frame(self.window)
        self.read_dicom_button = tk.Button(self.frame1, text="Read DICOM files", command=self.load_dicom_images)
        self.crop_frame = CropFrame(self.frame1, update_callback=self.update_mask_scale_preview)
        
        # Arranging widgets in frame1 
        self.read_dicom_button.grid(row=0, column=0, sticky='ew')
        self.crop_frame.grid(row=1, column=0, sticky='ew')
        self.frame1.grid_columnconfigure(0, weight=1)
        
        # Creating frame2 for mask preview
        self.frame2 = tk.Frame(self.window)
        self.preview_mask_label = tk.Label(self.frame2, text='Preview mask')
        self.mask_scale_preview = CreateMaskFrame(self.frame2, length=400, images=self.initial_images , update_callback=self.update_segmentation_view)

        # Arranging widgets in frame2
        self.preview_mask_label.grid(row=0, column=0) 
        self.mask_scale_preview.grid(row=2, column=0)

        # Creating frame3 for mask visualization
        self.frame3 = tk.Frame(self.window)
        self.view_mask_label = tk.Label(self.frame3, text='View mask')
        self.mask_view = SegmentationFrame(self.frame3, update_callback=self.update_result_view)

        # Arranging widgets in frame3
        self.view_mask_label.grid(row=0, column=0) 
        self.mask_view.grid(row=2, column=0)
        
        # Creating frame4 for viewing results
        self.frame4 = tk.Frame(self.window)
        self.view_results_label = tk.Label(self.frame4, text='View results')
        self.result_view = ResultFrame(self.frame4)
        self.sync_var = tk.IntVar()
        self.sync_button = tk.Checkbutton(self.frame4, text   ="Sync frames", variable=self.sync_var, command=self.toggle_sync)

        # Arranging widgets in frame4
        self.view_results_label.grid(row=0, column=0) 
        self.result_view.grid(row=1, column=0)
        self.sync_button.grid(row=2, column=0, sticky='n')

        # Arranging frames frame1, frame2, frame3, frame4
        self.frame1.grid(row=0, column=0, sticky='n')
        self.frame2.grid(row=0, column=1, sticky='n')
        self.frame3.grid(row=0, column=2, sticky='n')
        self.frame4.grid(row=0, column=3, sticky='n')
        
    def run(self):
        self.window.mainloop()
        
    def load_dicom_images(self):
        folder_path = filedialog.askdirectory()
        self.initial_images = utils.load_scan(folder_path)
        self.crop_frame.load_images(self.initial_images)
        self.mask_scale_preview.load_images(self.initial_images)
        self.mask_view.original_images = self.initial_images

    def update_mask_scale_preview(self, imgs, scale1_val, scale2_val, scale3_val, scale4_val, action):
        self.mask_scale_preview.load_images(imgs)
        self.mask_view.update_scale_values(scale1_val, scale2_val, scale3_val, scale4_val, action)
    
    def update_segmentation_view(self, masked_imgs=None, original_imgs=None):
        if masked_imgs is not None and original_imgs is not None:
    
            self.mask_view.load_images(masked_imgs, original_imgs)
        else:
            print("No images or masks received for update.")
    
    def update_result_view(self, imagini_contur, imagine_cerc, masti_reconstruite):
        self.result_view.load_images(imagini_contur, imagine_cerc, masti_reconstruite)
        
    def toggle_sync(self):
        if self.sync_var.get() == 1:
            if len(self.crop_frame.images) > 0 and len(self.mask_scale_preview.images) > 0:
                
                # Synchronization for mask_scale_preview
                self.mask_scale_preview.label.counter = self.crop_frame.label.counter
                self.mask_scale_preview.label.label_var.set(f"{self.mask_scale_preview.label.counter + 1} / {self.mask_scale_preview.label.max_count}")
                self.mask_scale_preview.update_image()
                
                # Setup buttons for synchronization
                self.crop_frame.button_next.config(command=self.increase_both)
                self.crop_frame.button_previous.config(command=self.decrease_both)
                self.mask_scale_preview.button_next.config(command=self.increase_both)
                self.mask_scale_preview.button_previous.config(command=self.decrease_both)

            if len(self.mask_view.images) > 0:
                
                # Synchronization for mask_view
                self.mask_view.label.counter = self.crop_frame.label.counter
                self.mask_view.label.label_var.set(f"{self.mask_view.label.counter + 1} / {self.mask_view.label.max_count}")
                self.mask_view.update_image()
                
                # Setup buttons for synchronization
                self.mask_view.button_next.config(command=self.increase_both)
                self.mask_view.button_previous.config(command=self.decrease_both)

            if len(self.result_view.circle_images) > 0:
                # Synchronization for result_view
                self.result_view.label.counter = self.crop_frame.label.counter
                self.result_view.label.label_var.set(f"{self.result_view.label.counter + 1} / {self.result_view.label.max_count}")
                self.result_view.update_image()
                
                # Setup buttons for synchronization
                self.result_view.button_next.config(command=self.increase_both)
                self.result_view.button_previous.config(command=self.decrease_both)

        else:
            # Reset button commands to individual increase/decrease functions
            self.crop_frame.button_next.config(command=self.crop_frame.label.increase)
            self.crop_frame.button_previous.config(command=self.crop_frame.label.decrease)
            self.mask_scale_preview.button_next.config(command=self.mask_scale_preview.label.increase)
            self.mask_scale_preview.button_previous.config(command=self.mask_scale_preview.label.decrease)
            self.mask_view.button_next.config(command=self.mask_view.label.increase)
            self.mask_view.button_previous.config(command=self.mask_view.label.decrease)
            self.result_view.button_next.config(command=self.result_view.label.increase)
            self.result_view.button_previous.config(command=self.result_view.label.decrease)
    
    def increase_both(self):
        # Increase counters for all frames simultaneously
        self.crop_frame.label.increase()
        self.mask_scale_preview.label.increase()
        self.mask_view.label.increase()
        self.result_view.label.increase()

    def decrease_both(self):
        # Decrease counters for all frames simultaneously
        self.crop_frame.label.decrease()
        self.mask_scale_preview.label.decrease()
        self.mask_view.label.decrease()
        self.result_view.label.decrease()
        
        
app = MyApplication()
app.run()
