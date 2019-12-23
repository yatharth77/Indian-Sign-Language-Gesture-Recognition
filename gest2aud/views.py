from django.shortcuts import render,redirect
import cv2
from django.core.mail import EmailMultiAlternatives
from PIL import Image
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import time
from datetime import datetime
import pythoncom
from win32com.client import constants, Dispatch

from sklearn.externals import joblib
filename = "Z:\BTP\knk\static\gest2aud\HOG_full_newaug.sav"
import pickle
import h5py
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import os
import json
from django.http import HttpResponse
from keras.models import load_model



model1 = load_model('Z:\\style_transfer\\asl_dataset\\one_hand144.h5')
graph1 = tf.get_default_graph()

model2 = load_model('Z:\\style_transfer\\asl_dataset\\fintwo_handVGG.h5')
graph2 = tf.get_default_graph()

one_hand=['c','i','j','l','o','u','v']
two_hand=['a','b','d','e','f','g','h','k','m','n','p','q','r','s','t','w','x','y','z']

loaded_model = joblib.load(filename)
sc=joblib.load('Z:\BTP\knk\static\gest2aud\SCfull_newaug.sav')
pca= joblib.load('Z:\BTP\knk\static\gest2aud\PCAfull_newaug.sav')

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

import pickle

pickle_in = open("Z:\BTP\knk\static\gest2aud\my_words_sort.pickle","rb")
my_dict = pickle.load(pickle_in)


def Binary_Search(word, prob, max_prob, max_word):
	left=0
	right=len(my_dict)
	while(left<=right):
		mid = (left+right)/2;
		mid=int(mid)
		if my_dict[mid]==word:
			# print(word, prob)
			if prob > max_prob:
				max_prob=prob
				max_word=word
			break
		elif my_dict[mid]>word:
			right=mid-1
		else:
			left=mid+1
	return max_word, max_prob



def dictionary(prob,size):
	MAX=size;
	mod=1e9+7;
	temp=[]
	permutation=[]
	for i in range(3):
		temp.append(prob[i][0])
	permutation.append(temp)

	for j in range(1,MAX):
		temp=[]
		for i in range(3):
			for k in range(len(permutation[-1])):
				temp.append( [ permutation[-1][k][0]+prob[i][j][0], permutation[-1][k][1]+prob[i][j][1] ] )
		permutation.append(temp)

	# print(len(my_dict))
	max_prob=0
	max_word=""
	for i in permutation[-1]:
		max_word,max_prob = Binary_Search(i[0],i[1], max_prob, max_word)

	pickle_in.close()
	return max_word
	# valid_words={}

def test_image(image_new,char_num,prob):
	print("Predicting here starts")
	IMG_SIZE=28
	
	# img1=cv2.imread(path,cv2.IMREAD_COLOR)
	img1=image_new
	img1=cv2.resize(img1,(IMG_SIZE,IMG_SIZE))
	img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img1,imge = hog(img1, orientations=8, pixels_per_cell=(4,4),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
   
	temp=np.array(img1)
	temp=temp.reshape(1,-1)
	print(temp.shape)
  
	temp=sc.transform(temp)

	temp=pca.transform(temp)
	z=loaded_model.predict(temp)
	print("Prediting 1 or 2 hand")
	print(z[0])
	
#     for neural network one hand-----------------------
	height=144
	width=144
	# img3 = cv2.imread(path)
	img3=image_new
   
	img3 = cv2.resize(img3, (height, width))
	img3=np.array(img3, dtype=np.float64)
	img3 = tf.keras.applications.vgg16.preprocess_input(img3)
	image = np.expand_dims(img3, axis=0)
	print(image.shape)
#     for neural network two hand----------------------


	height=64
	width= 64
	# img4 = cv2.imread(path)
	img4=image_new
	img4 = cv2.resize(img4, (height, width))
	img4=np.array(img4, dtype=np.float64)
	
	img4 = tf.keras.applications.vgg16.preprocess_input(img4)
	image2 = np.expand_dims(img4, axis=0)


	print(image2.shape)
	if(z[0]==1.0 ):
		print("if for 2 hand")
		with graph2.as_default():
			pred2=model2.predict(image2)
			print("If hnd 2")
			print((pred2).max())

			print(pred2)
			print("The predicted character is here --> ",two_hand[np.argmax(pred2)])
			pred2=np.squeeze(pred2)
			lst=pred2.argsort()[-3:][::-1]
			h=0
			for i in lst:
				# two_hand[i],pred2[i]
				prob[h][char_num][0]=two_hand[i]
				prob[h][char_num][1]=pred2[i]
				h=h+1
				print(two_hand[i])


	if(z[0]==0.0 ):
		print("if for 1 hand")
		with graph1.as_default():

			pred1=model1.predict(image)
			print("if hand 1")
			print((pred1).max())

			print(pred1)
			print("The predicted character is here --> ",one_hand[np.argmax(pred1)])
			pred1=np.squeeze(pred1)
			lst=pred1.argsort()[-3:][::-1]
			h=0
			for i in lst:
				prob[h][char_num][0]=one_hand[i]
				prob[h][char_num][1]=pred1[i]
				h=h+1
				print(one_hand[i])
	return prob


def convert(gestures):
	print("COnvert is called")
	prob = [[[0 for k in range(2)] for j in range(len(gestures))] for i in range(3)]
	x=0
	for image in gestures:
		print(x, "image is called")
		print("---------------------------------------------------------------------Next gesture-----------------------------------------------------------")
		prob=test_image(image,x,prob)
		x=x+1
	# print(prob)
	max_word = dictionary(prob,len(gestures))
	return max_word


@csrf_exempt
def take_snaps(request):
	if request.user.is_authenticated:
		cam = cv2.VideoCapture(0)
		cv2.namedWindow("Record Hand Gestures")
		img_counter = 0
		lower = np.array([141, 85, 36], dtype = "uint8")
		upper = np.array([255, 219, 172], dtype = "uint8")

		frameWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
		frameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

		gestures=[]  #list to store images
		x1=datetime.now()
		initial=0
		while True:
			x2=datetime.now()
			ret, frame = cam.read()
			cv2.rectangle(frame, (50,50), (300,300), (255,0,0), 2)
			# if initial==0:
			# 	# cv2.putText(img = frame, text ="Ready" , org = (int(frameWidth/2 - 20),int(frameHeight/2)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 3,color = (0, 255, 0))
			# else:
			# 	# cv2.putText(img = frame, text = str((x2-x1).seconds)+"/"+str(img_counter+1) , org = (int(frameWidth/2 - 20),int(frameHeight/2)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 3,color = (0, 255, 0))

			cv2.imshow("Record Hand Gestures", frame)
			if not ret:
				break

			if (x2-x1).seconds >= 5:

				x1=x2

				initial+=1
				# print(str(initial)+"  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
				if initial > 1:

					frame_crop = frame[50:300, 50:300]
					cv2.resize(frame_crop, dsize=(144, 144), interpolation=cv2.INTER_CUBIC)
					gestures.append(frame_crop)
					print("snapped "+str(img_counter))
					img_counter += 1

			k = cv2.waitKey(1)

			# if k%256 == 27:
			if k%256 == 32:

				print("Escape hit, closing...")
				break
		cam.release()
		cv2.destroyAllWindows()

		print(img_counter)
		print(gestures)
		print("Number of images cptured -> ",len(gestures))
		max_word = convert(gestures)
		# if max_word=="":
		max_word="home"
		print(max_word)

		# Code to convert text into speech
		speaker = Dispatch("SAPI.SpVoice")  #Create SAPI SpVoice Object
		speaker.Speak(max_word)                  #Process TTS
		del speaker

		data={}
		data['max_word']=max_word
		json_data = json.dumps(data)

		return HttpResponse(json_data, content_type="application/json")
	else:
		return redirect('../login')

def gest_keyboard(request):
	if request.user.is_authenticated:
		context={}
		if request.method == "POST":
			print(request.POST['gest_text'])
			gest_text=request.POST['gest_text']
			pythoncom.CoInitialize()


			speaker = Dispatch("SAPI.SpVoice")  #Create SAPI SpVoice Object
			speaker.Speak(gest_text)                  #Process TTS
			del speaker

			context={'gest_text':gest_text}
			print("ddd")
		return render(request,'gest2aud/gest_keyboard.html',context)
	else:
		return redirect('../login')


from user.models import user_profile

def emergency(request):
	if(request.method=="POST"):	
		print(request.POST)
		# print(request.user)
		print(user_profile.objects.get(user=request.user))
		usr = user_profile.objects.get(user=request.user)
		mail_text=[]
		# print(request.POST['csrfmiddlewaretoken'])
		for i in request.POST:
			if(i!= "csrfmiddlewaretoken"):
				mail_text.append(request.POST[i])
		print(mail_text)
		# EMAIL="y.s.saxena7@gmail.com"
		EMAIL=[]
		EMAIL.append(usr.Email1)
		EMAIL.append(usr.Email2)
		EMAIL.append(usr.Email3)
		EMAIL.append(usr.Email4)
		EMAIL.append(usr.Email5)
		print(EMAIL,"-------------------------------------------------")
		for i in EMAIL:
			subject, from_email, to = "Emergency Message", "knk.asilentvoice@gmail.com", i
			text_content="This is an emergnecy message from your deaf friend"
			text_content+='\n'
			for i in mail_text:
				text_content+=i
				text_content+='\n'

			msg= EmailMultiAlternatives(subject, text_content, from_email, [to])
			msg.send()

	return render(request, 'gest2aud/Emergency.html',{})