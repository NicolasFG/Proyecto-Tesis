import tensorflow as tf
from tensorflow import keras
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.models.resnet import ResNet101RetinaNet
from keras import layers
from keras_retinanet.models import default_classification_model, default_regression_model

import numpy as np
import os


current_path = os.getcwd()

# Rutas de los archivos de entrenamiento y validación
train_annotations =  current_path + '\\entrenamiento.csv'
train_images_dir = current_path + '\\entrenamiento'
val_annotations = current_path + '\\validacion.csv'
val_images_dir = current_path + '\\validacion'

# Crear generador de datos para entrenamiento
train_generator = CSVGenerator(train_annotations, train_images_dir)

# Crear generador de datos para validación
val_generator = CSVGenerator(val_annotations, val_images_dir)



# Cargar modelo pre-entrenado (por ejemplo, ResNet50)
#base_model = models.backbone('resnet50').retinanet(num_classes=train_generator.num_classes())
#base_model = models.backbone('ResNet-101').retinanet(num_classes=train_generator.num_classes())

#base_model = models.backbone('ResNext-101').retinanet(num_classes=train_generator.num_classes())
base_model = ResNet101RetinaNet(num_classes=train_generator.num_classes())



# Agregar capas dilatadas a la arquitectura base
input_layer = base_model.input
resnet_layers = base_model.layers[1].layers

# Encuentra la última capa convolucional de ResNet-101
last_conv_layer = None
for layer in resnet_layers[::-1]:
    if isinstance(layer, layers.Conv2D):
        last_conv_layer = layer
        break

# Agrega capas dilatadas después de la última capa convolucional
dilated_layers = layers.Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu')(last_conv_layer.output)
dilated_layers = layers.Dropout(0.5)(dilated_layers)
dilated_layers = layers.Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu')(dilated_layers)
dilated_layers = layers.Dropout(0.5)(dilated_layers)

# Construye el modelo completo con las capas dilatadas
classification_model = default_classification_model(dilated_layers, num_classes=train_generator.num_classes())
regression_model = default_regression_model(dilated_layers, num_anchors=classification_model.output_shape[1] // train_generator.num_classes())

model = models.retinanet_bbox(inputs=input_layer, num_classes=train_generator.num_classes(), backbone=dilated_layers, 
                              regression_model=regression_model, classification_model=classification_model)


# Añadir conexiones laterales
lateral_layers = [dilated_layers]
for i in range(len(resnet_layers) - 2, -1, -1):
    layer = resnet_layers[i]
    if isinstance(layer, layers.Conv2D):
        upsampled = layers.UpSampling2D(size=(2, 2))(lateral_layers[-1])
        lateral = layers.Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(layer.output)
        lateral_layers.append(layers.add([upsampled, lateral]))

classification_model = default_classification_model(lateral_layers[-1], num_classes=train_generator.num_classes())
regression_model = default_regression_model(lateral_layers[-1], num_anchors=classification_model.output_shape[1] // train_generator.num_classes())

model = models.retinanet_bbox(inputs=input_layer, num_classes=train_generator.num_classes(), backbone=dilated_layers,
                              regression_model=regression_model, classification_model=classification_model)


# Fusión de características
fused_features = layers.concatenate(lateral_layers, axis=-1)

classification_model = default_classification_model(fused_features, num_classes=train_generator.num_classes())
regression_model = default_regression_model(fused_features, num_anchors=classification_model.output_shape[1] // train_generator.num_classes())

model = models.retinanet_bbox(inputs=input_layer, num_classes=train_generator.num_classes(), backbone=fused_features,
                              regression_model=regression_model, classification_model=classification_model)


# Ajustar los parámetros de los anclajes
anchor_parameters = models.default_anchor_parameters()
anchor_parameters.sizes = anchor_parameters.sizes._replace(
    base=16, ratios=[0.5, 1, 2], scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
anchor_parameters = anchor_parameters._replace(
    sizes=anchor_parameters.sizes, strides=anchor_parameters.strides, ratios=anchor_parameters.ratios)

# Ajustar la estructura de la pirámide de características
model._build_pyramid(model.inputs, model.outputs, anchor_parameters)

# Agregar nuevos niveles específicos para detección de objetos pequeños
model.add_extra_pyramid_level(64, (3, 3), strides=(2, 2), padding='same', name='small_object_p3')
model.add_extra_pyramid_level(64, (3, 3), strides=(2, 2), padding='same', name='small_object_p4')
model.add_extra_pyramid_level(64, (3, 3), strides=(2, 2), padding='same', name='small_object_p5')


# Crear el modelo completo
#model = models.retinanet_bbox(inputs=base_model.inputs, outputs=base_model.outputs, anchor_parameters=anchor_parameters)

# Compilar el modelo
model.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss={'regression': keras.losses.Huber(), 'classification': keras.losses.BinaryCrossentropy()}, metrics=['accuracy'])

#Entrenar el modelo
model.fit(train_generator, validation_data=val_generator, epochs=100)
