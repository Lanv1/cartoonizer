import cv2
import random
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import vgg19


def convertToHSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def imgToHSV_asGray(path, filename):
    print(path)
    print(filename)
    print(path + filename )
    img = cv2.imread(path + filename + ".jpg")
    image = convertToHSV(img)
    (h, s, v) = cv2.split(image)
    image2 = cv2.merge([h, h, h])
    # cv2.imwrite(path+"HSV/"+filename+".jpg", image2)
    # return path+"HSV/"+filename+".jpg"
    return image2


def HSV_asGreyToRGB(HSVgray, hsv):
    #    h = cv2.cvtColor(HSVgray, cv2.COLOR_BGR2GRAY)
    (h1, h2, h3) = cv2.split(HSVgray)
    h = cv2.addWeighted(cv2.addWeighted(h1, 0.33, h2, 0.33, 0), 0.66, h3, 0.33, 0.01)
    (_, s, v) = cv2.split(hsv)
    img = cv2.merge([h, s, v])
    return convertToRGB(img)


print("module chargé")

filename = "faker"
styleName = "oggy"
# styleName = "nuage"
# styleName = "nuit-etoilee"
# outputPath = "../Images/Outputs/transfert_de_style/blanc/"
outFile = "faker_oggy_style/"
outputPath = "asset/" + outFile

# image_path = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
path = "asset/"
imagePath = "asset/" + filename + ".jpg"
imgOriginal = cv2.imread(imagePath)
imageHSV = imgToHSV_asGray(path, filename)
# imgCV2 = cv2.imread (image_path)
stylePath = "asset/" + styleName + ".jpg"

# style_reference_image_path
styleHSV = imgToHSV_asGray(path, styleName)

result_prefix = filename + "_" + styleName + "_generated"

# Weights of the different loss components
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# Dimension de l'image généré
width, height = keras.preprocessing.image.load_img(imagePath).size
# img_nrows = 400
# img_ncols = int(width * img_nrows / height)
img_nrows = height
img_ncols = width

# affiche image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# read the image file in a numpy array
a = plt.imread(imagePath)
b = plt.imread(stylePath)
f, axarr = plt.subplots(1, 2, figsize=(15, 15))
axarr[0].imshow(a)
axarr[1].imshow(b)
plt.show()


# (h, s, v) = cv2.split(newImg)


def preprocess_image(img):
    # Util function to open, resize and format pictures into appropriate tensors

    # img = keras.preprocessing.image.load_img(
    #    image_path, target_size=(img_nrows, img_ncols))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img = keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


# utiliser pour calculer la perte de style
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# maintient l'image générée proche des textures locales de l'image de style
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


# maintient la représentation de haut niveau de l'image générée proche de l'image de base
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


## premières couches -> granularité (style)
## dernières couches -> structule de l'image


def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

# The layer to use for the content loss.
content_layer_name = "block5_conv2"

content_weight = 2.5e-8
style_weight = 1e-6


def compute_loss(combination_image, base_image, style_reference_image):
    # 1. Combine all the images in the same tensioner.
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )

    # 2. Get the values in all the layers for the three images.
    features = feature_extractor(input_tensor)

    # 3. Inicializar the loss

    loss = tf.zeros(shape=())

    # 4. Extract the content layers + content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    # print(base_image_features)
    combination_features = layer_features[2, :, :, :]

    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # 5. Extraer the style layers + style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

        # add var loss ?
    return loss


# décorateur pour compiler la fonction et la rendre plus rapide
@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in
# VGG19 (as a dict).
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

base_image = preprocess_image(imageHSV)
style_reference_image = preprocess_image(styleHSV)
combination_image = tf.Variable(preprocess_image(imageHSV))

iterations = 4000
for i in range(1, iterations + 1):
    print("iteration :    ", i, "   \n")
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 5 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        HSVgray = deprocess_image(combination_image.numpy())
        fname = outputPath + result_prefix + "_at_iteration_%d.png" % i
        keras.preprocessing.image.save_img(fname, HSV_asGreyToRGB(HSVgray, imageHSV))
