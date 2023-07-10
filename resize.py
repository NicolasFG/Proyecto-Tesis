from skimage import io, color, transform, util, filters
import numpy as np
import glob
import os

current_path = os.getcwd()

def resize_img(path, nuevo_dataset):
    
    os.makedirs(nuevo_dataset, exist_ok=True)

    # Itero en mis imagenes
    for filename in os.listdir(path):
       
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Leo la imagen
            img_path = os.path.join(path, filename)
            img = io.imread(img_path)

            # Aplico el resize 300x600
            resized_image = transform.resize(img, (300, 600))

             # Convierte el tipo de datos a uint8
            resized_image = util.img_as_ubyte(resized_image)

            # Guardo mi nueva imagen
            output_path = os.path.join(nuevo_dataset, filename)

            io.imsave(output_path, resized_image)


input_path = current_path + "\\dataset"
output_path = current_path + "\\nuevo_dataset"
resize_img(input_path, output_path)