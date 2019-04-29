import os
import cv2
import pickle
from PIL import Image
import numpy as np
image_dir ="C:\\Users\\Omen\\Desktop\\AI\\Project\\images"

face_cascade=cv2.CascadeClassifier('C:\\Users\\Omen\\Desktop\\AI\\Project\\haarcascade_frontalface_alt2.xml')
recognizer =cv2.face.LBPHFaceRecognizer_create()
x_train = []
y_labels =[]
current_id=0
label_ids={}
for root ,dirs ,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("pgm"):
            path=os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            if not label in label_ids:
                label_ids[label]=current_id
                current_id +=1
            id_= label_ids[label]
            pil_image=Image.open(path).convert("L")
            pil_image = pil_image.resize((1000,1000), Image.ANTIALIAS)
            image_array=np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array)
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")

            
