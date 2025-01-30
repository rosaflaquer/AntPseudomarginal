#%%
import os
import cv2
import numpy as np
import pandas as pd

def load_images_from_dirs(directories,filename):
    images = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        name_chars = len(filename)
        files = [f for f in os.listdir(directory) if f[:name_chars]==filename]
        if not files:
            print(f"No images found in directory: {directory}")
            continue
        
        img_path = os.path.join(directory, files[0])  # Load the first image found
        img = cv2.imread(img_path)
        if img is not None:
            images.append((img, files[0]))  # Store image with its filename
    
    return images

def create_collage(images, imagenames, grid_size=(2, 2),  output_path="collage.jpg"):  
    if not images:
        print("No images to create a collage.")
        return
    
    min_height = min(img[0].shape[0] for img in images)
    min_width = min(img[0].shape[1] for img in images)
    
    images_resized = [(cv2.resize(img[0], (min_width, min_height)), img[1]) for img in images]
    
    grid_rows, grid_cols = grid_size
    collage = np.zeros((grid_rows * min_height, grid_cols * min_width, 3), dtype=np.uint8)
    
    idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (0, 0, 0)
    thickness = 3
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx < len(images_resized):
                img = images_resized[idx][0]
                img_with_text = img.copy()
                cv2.putText(img_with_text, imagenames[idx], (80, 80), font, font_scale, font_color, thickness, cv2.LINE_AA)
                collage[r * min_height: (r + 1) * min_height, c * min_width: (c + 1) * min_width] = img_with_text
                idx += 1
    
    cv2.imwrite(output_path, collage)
    print(f"Collage saved at {output_path}")
#%%

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
proj_path = os.path.split(proj_path)[0] 
traj_dir = os.path.join(proj_path,"Data","Ant_data")
data_file = "2022_Transformed_width_50-frames_40.dat"
datadf = pd.read_csv(os.path.join(traj_dir,data_file))
main_dir = os.path.join(proj_path,"Data","Fits","Fitsnomin","mcmc","Data","Fits")
folders = os.listdir(main_dir)
converged = []
filename = "Distr_sigma"
directory_names = os.listdir(main_dir)
directories = [os.path.join(main_dir,i) for i in directory_names]

images = load_images_from_dirs(directories,filename)
grid_size = (33, 5)  # Adjust grid size based on number of images
create_collage(images,directory_names,grid_size,output_path=filename+".jpg")

# %%
