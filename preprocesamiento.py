from skimage import io, color, transform, util, filters
import numpy as np
import glob
import os

current_path = os.getcwd()

def resize_img(path, output_folder):
    # Create el folder si este no existe
    os.makedirs(output_folder, exist_ok=True)

    # Itero en mis imagenes
    for filename in os.listdir(path):
       
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Leo la imagen
            img_path = os.path.join(path, filename)
            img = io.imread(img_path)

            # Aplico el resize
            resized_image = transform.resize(img, (200, 200))

            # Guardo mi nueva imagen
            output_path = os.path.join(output_folder, filename)
            io.imsave(output_path, resized_image)

# Example usage
input_path = current_path + "/Dataset"
output_path = current_path + "/output_folder"
#resize_img(input_path, output_path)



# Carga las imágenes
paths = glob.glob('ruta/a/tu/dataset/*')
images = [io.imread(path) for path in paths]

processed_images = []

for image in images:
    
    # Convertir a escala de grises
    image = color.rgb2gray(image)

    # Volteo horizontal (puedes cambiar a 'vertical' para volteo vertical)
    image = np.fliplr(image)

    # Ruido
    image = util.random_noise(image)

    # Rotación
    image = transform.rotate(image, 90) 

    # Desenfoque
    image = filters.gaussian(image, sigma=1)  

    processed_images.append(image)

# Ahora, processed_images contiene tus imágenes procesadas
