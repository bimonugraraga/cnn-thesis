#Import Library 
import tensorflow as tf
import numpy as np
import cv2
import RPi.GPIO as GPIO
import time

#Membuat fungsi untuk mengatur solenoid
def buka(pin):
	GPIO.output(pin, GPIO.HIGH)

#Mengakses Kamera	
cap = cv2.VideoCapture(0)
KATEGORI = ["Medis", "Non Medis"]

#Mengakses model training
model = tf.keras.models.load_model("protoFix.model")

#Pengolahan Citra dan Ekstraksi Fitur
while True :
	success, img = cap.read()
	start = time.time()
	img = cv2.resize(img, (512,512))
	
	ycbcr = img.copy()
	ycbcr = cv2.cvtColor(ycbcr, cv2.COLOR_BGR2YCR_CB)
	
	lower_y = np.array([0, 0, 118])
	upper_y = np.array([255, 255, 255])
	
	masky = cv2.inRange(ycbcr, lower_y, upper_y)
	resulty = cv2.bitwise_and(ycbcr, ycbcr, mask = masky)
	
	resulty2 = resulty.reshape(-1, 512, 512, 3)

#Melakukan prediksi kelas
	prediction = model.predict([resulty2])

#Membuka dan Mengunci Solenoid Lock
	if prediction >= 0.9999998 :
		print("Jenis Masker	: Non medis", prediction)
		end1 = time.time()
		print("Waktu	:", end1 - start)
		pass
	if prediction < 0.9999998 :
		print("Jenis Masker	: Medis", prediction)
		end2 = time.time()
		print("Waktu	:", end2 - start)
		GPIO.setmode(GPIO.BCM)
		GPIO.setup(21, GPIO.OUT)
		buka(21)
		print("Buka")
		time.sleep(3)
		GPIO.cleanup()
		pass

#Menampilkan Citra	
	print("prediksi:", KATEGORI[int(prediction[0][0])], prediction)
	cv2.imshow("img", resulty)
	
	if cv2.waitKey(5000) & 0xff==ord('q') :
		break
