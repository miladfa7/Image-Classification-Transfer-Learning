# Image-Classification-Transfer-Learning
### Building ResNet152V2 Model for Image Classification with Small Dataset (99.5% accuracy)

Number of classes: 20 (Classes 0-19)<br>

<font color="red"> Classes = owl | galaxy | lightning | wine-bottle | t-shirt | waterfall |  sword |  school-bus |
                         calculator | sheet-music | airplanes |  lightbulb |  skyscraper | mountain-bike | fireworks | 
                         computer-monitor | bear | grand-piano | kangaroo | laptop ]</font><br>
<br>                     
<b>Dataset Structure</b><br>
   Two folders:<br>
  Training: 1554 images<br>
    Test: 500 images<br>
 
    Images per class:
    
    school-bus : 73
    laptop : 100
    t-shirt : 100
    grand-piano : 70
    waterfall : 70
    galaxy : 56
    mountain-bike : 57
    sword : 77
    wine-bottle : 76
    owl : 95
    fireworks : 75
    calculator : 75
    sheet-music : 59
    lightbulb : 67
    bear : 77
    computer-monitor : 100
    airplanes : 100
    skyscraper : 70
    lightning : 100
    kangaroo : 57   

### visualization of training data 
<img src="https://github.com/miladfa7/Image-Classification-Transfer-Learning/blob/master/images/dataet%20image%20classification.png" width="500" alt="image classification with transfer learning ">

### Result 
The accuracy of the training reached 99.5% in 50 epoch.<br>
The accuracy of the test reached 95% that i submitted to kaggle.<br>

<img src="https://github.com/miladfa7/Image-Classification-Transfer-Learning/blob/master/images/result.png" width="800" alt="result resnet152"> </img>


### CSV file for kaggle submission<br>
```
predicted_class_indices=np.argmax(pred,axis=1)
labels = train_gen.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [k for k in predicted_class_indices]

filenames=test_gen.filenames
FN=[]
for i in filenames:
  f = i[5:]
  FN.append(f)
 
results=pd.DataFrame({"Id":FN,
                      "Category":predictions})
results.to_csv("submission.csv",index=False)
```
