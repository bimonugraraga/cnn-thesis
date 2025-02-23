#Import Library
import numpy as np
import os
import cv2
import pickle
import random

#Akses Dataset
DATADIR = "Masker3"
CATAGORIES = ["Medis", "Non Medis"]

IMG_SIZE = 512
training_data =[]
#Memberikan Label
def create_training_data():
	for category in CATAGORIES :
		path = os.path.join(DATADIR,category)
		class_num = CATAGORIES.index(category)
		for img in os.listdir(path):
			try :
#Pengolahan Citra				
				image = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
				image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
			
				ycbcr = image.copy()
				ycbcr = cv2.cvtColor(ycbcr, cv2.COLOR_BGR2YCR_CB)

				lower_y = np.array([0, 0, 118])
				upper_y = np.array([255, 255, 255])

				mask = cv2.inRange(ycbcr, lower_y, upper_y)
				result = cv2.bitwise_and(ycbcr,ycbcr,mask = mask)

				result1 = np.array(result)
				training_data.append([result, class_num])
			except Exception as e :
				pass
create_training_data()
print(len(training_data))
random.shuffle(training_data)

#Menyimpan Hasil Ekstraksi Fitur
train_X = []
label_y = []

for features, label in training_data :
	train_X.append(features)
	label_y.append(label)

train_X = np.array(train_X).reshape(-1, 512, 512, 3)

pickle_out = open("train_XFix.pickle", "wb")
pickle.dump(train_X, pickle_out)
pickle_out.close()

pickle_out = open("label_yFix.pickle", "wb")
pickle.dump(label_y, pickle_out)
pickle_out.close()

print("sukses")
