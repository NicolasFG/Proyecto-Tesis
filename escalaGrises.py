from skimage import io, color, transform, util, filters
import numpy as np
import glob
import os

current_path = os.getcwd()

def escala_grisesimg(path, nuevo_dataset):

    os.makedirs(nuevo_dataset, exist_ok=True)

    for filename in os.listdir(path):
       
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):

            img_path = os.path.join(path, filename)
            img = io.imread(img_path)

            img = color.rgb2gray(img)

            resized_image = util.img_as_ubyte(img)

            output_path = os.path.join(nuevo_dataset, filename)

            io.imsave(output_path, resized_image)


input_path = current_path + "\\dataset"
output_path = current_path + "\\nuevo_dataset"
escala_grisesimg(input_path, output_path)