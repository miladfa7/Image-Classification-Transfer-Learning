import tensorflow 
from PIL import Image
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

train_aug = os.path.join('/content/drive/My Drive/Colab Notebooks/Dataset/', 'training')
valid_aug = os.path.join('/content/drive/My Drive/Colab Notebooks/Dataset/', 'validation')

train_cln_aug = [x[1] for x in os.walk(train_aug)]
train_cln_aug = train_cln_aug[0]

valid_cln_aug = [x[1] for x in os.walk(valid_dir)]
valid_cln_aug = valid_cln_aug[0]

 
for tcg in train_cln_aug:
  
  save_to_dir = '/content/drive/My Drive/Colab Notebooks/Dataset/training/' + tcg
  filename =  save_to_dir +'/*.jpg'
  im=Image.open(filename)
  image_list.append(im)

  image = image_list[0]
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  imageGen = aug.flow(image, batch_size=1, save_to_dir=save_to_dir ,save_prefix="image", save_format="jpg")
  
  # generating 10 sample for each training image
  total = 0
  final = 10
  for image in imageGen:
    
    total += 1
    if total ==final :
      break

for vcg in train_cln_aug:
  
  save_to_dir = '/content/drive/My Drive/Colab Notebooks/Dataset/validation/' + vcg
  filename =  save_to_dir +'/*.jpg'
  im=Image.open(filename)
  image_list.append(im)

  image = image_list[0]
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  imageGen = aug.flow(image, batch_size=1, save_to_dir=save_to_dir ,save_prefix="image", save_format="jpg")
  
  # generating  2 sample for each validation image
  total = 0
  final = 2
  for image in imageGen:
    total += 1
    if total ==final :
      break