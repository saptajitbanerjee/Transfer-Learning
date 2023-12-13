import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as xet

### READ DATA ###

df = pd.read_csv('./labels.csv')

filename = df['filepath'][0]

def getFileName(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./images_labeled',filename_image)
    return filepath_image

image_path = list(df['filepath'].apply(getFileName))
"""
### Verify Image and Output ###

file_path = image_path[0]
img = cv2.imread(file_path)
cv2.namedWindow('example',cv2.WINDOW_NORMAL)
cv2.imshow('example',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.rectangle(img)
"""

### Data Preprocessing ###
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img,img_to_array

labels = df.iloc[:,1:].values
data = []
output= []

for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    print(img_arr)
    h,w,d = img_arr.shape
    
    #Preprocessing
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_img_arr = load_image_arr/225.0
    
    #Normalization to Labels
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w, xmax/w
    nymin,nymax = ymin/w, ymax/w
    label_norm = (nxmin,nxmax,nymin,nymax)
    
    #Append
    data.append(norm_load_img_arr)
    output.append(label_norm)

X = np.array(data,dtype=np.float32)
y = np.array(data,dtype=np.float32)

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)


