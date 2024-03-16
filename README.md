# DICOM Bone Lesion Detection

## Description
This project aims to develop a solution for importing and visualizing batch of DICOM files (CT scan files), focusing on segmenting bone tissue and identifying potential osteolytic lesions (a type of bone cancer). Developed exclusively in Python, it leverages basic libraries such as Tkinter for the user interface, alongside specialized libraries for processing like skimage, sklearn, and numpy, to offer a comprehensive tool for medical image analysis.

## Technologies Used
- **User Interface:** Tkinter
- **Image Processing:** scikit-image (skimage), scikit-learn (sklearn), NumPy

## Project Structure
The project is structured into two main files and utility files for various processes:
- `back.py`: Contains classes that provide utility support for specific functionalities like image import, navigating through the image volume, formatting, masking, determining parameters for bone masking, and implementing image processing algorithms.
- `front.py`: Aims to interconnect the aforementioned files and implements mechanisms to determine the interactions between objects.
- Additional utility files are used for determining and parameterizing methods applied in bone and diseased tissue segmentation algorithms.

## Installation Instructions
1. Clone the repository to your local machine.
2. Ensure Python and all imported libraries are installed.
3. Run the `front.py` file to start the application, which should then be accessible for use.

## User Guide
The application is designed with four utility frames:
1. **Frame 1:** Used for reading CT DICOM files and includes cropping methods for better segmentation.
2. **Frame 2:** Applies tissue masking procedures, offering three methods: manual, k-means, and morphological contouring for bone tissue.
3. **Frame 3:** Implements the algorithm for detecting cancerous tissue using various image processing techniques.
4. **Frame 4:** Displays results with the diseased tissue highlighted and contoured, also allowing the user to export the results.

Each frame integrates a functionality for viewing the specific images returned by all applied methods, along with a system for navigating through the image batch, enabling synchronization across all frames.
