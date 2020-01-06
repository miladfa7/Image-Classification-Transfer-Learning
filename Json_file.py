import json
import os 
train_dir = os.path.join('/content/drive/My Drive/Colab Notebooks/myDataset/DT', 'training')
valid_dir = os.path.join('/content/drive/My Drive/Colab Notebooks/myDataset/DT', 'valid')

train_class_name = [x[1] for x in os.walk(train_dir)]
train_class_name = train_class_name[0]

valid_class_name = [x[1] for x in os.walk(valid_dir)]
valid_class_name = valid_class_name[0]

train_json = {}
valid_json = {}
 
for tcn in train_class_name:
  
  address =  '/content/drive/My Drive/Colab Notebooks/myDataset/DT/training/' + tcn
  for x in os.walk(address):

    train_json[tcn] = x[2] 
for vcn in valid_class_name:
  
  address =  '/content/drive/My Drive/Colab Notebooks/myDataset/DT/valid/' + vcn
  for x in os.walk(address):
    
    valid_json[vcn] = x[2] 
dataset_split = {
    'training':train_json,
    'validation':valid_json
}
with open('dataset_split.json', 'w') as fp:
  json.dump(dataset_split, fp)