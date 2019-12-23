from django.shortcuts import render,redirect
from django.http import HttpResponse
import speech_recognition as sr
from .forms import UploadAudio
from .models import AudioDb
from knk.settings import *
from django.core.files.base import ContentFile
import matplotlib
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from random import randint
from pydub import AudioSegment
from os import path
from pydub import AudioSegment
import json

# Create your views here.
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def filename(audio):
	return str(audio) , str(audio).split(".")[0]+'.txt' , str(audio).split(".")[0]+'.png'

def audio_url(audio):
	return 'aud2gest/audioFiles/'+str(audio)

def image_url(text,image_name):
	Alp = {}
	for code in range(ord('A'), ord('Z') + 1):
		Alp[chr(code)]=os.path.join(MEDIA_ROOT,"Alphabets",chr(code)+".jpg")
	words=text.split(' ')
	max_len=max(len(w) for w in words)
	if len(words) < 4:
		words+=['','','']
	plt.subplot()
	j=0
	for word in words:
			i=1+j*max_len
			for key in word:
				image = mpimg.imread((Alp[key.upper()]))
				plt.subplot(len(words),max_len,i)
				plt.axis('off')
				plt.imshow(image, aspect='auto')
				plt.subplots_adjust(left=0, right=1, top=1, bottom=0,hspace=0, wspace=0)
				i+=1
			j+=1 
	image_path = os.path.join(MEDIA_ROOT,"aud2gest/imageFiles",image_name)
	plt.savefig(image_path,figsize=(15,15))
	# plt.show(image_path)
	return image_path , 'aud2gest/imageFiles/'+image_name

def text_url(text,text_name):
	text_path=os.path.join(MEDIA_ROOT,"aud2gest/textFiles",text_name)
	file1=open(os.path.join(text_path),'w')
	file1.write(text)
	file1.close()
	return text_path , 'aud2gest/textFiles/'+text_name


def audio_to_text(audio_voice):
	r = sr.Recognizer()
	audio_name=str(audio_voice)
	audio_path=os.path.join(MEDIA_ROOT,"aud2gest/audioFiles",audio_name)
	text=""
	with sr.AudioFile(audio_path) as source:
		audio = r.record(source)
		print ('Done!')

	try:
		text = r.recognize_google(audio)
		print (text)
	except Exception as e:
		print (e)
	return text

@csrf_exempt
# @login_required
def home(request):
	if request.user.is_authenticated:
		if request.method=="POST":
			print("Brook was here")
			# print(request.FILES['choice'])
			instance = AudioDb()
			if "file" in request.FILES:
				audio=request.FILES["file"]
				instance.audiofile.save(audio.name, audio)
				instance.save()
			else:
				audio=request.session["filename"]
				audio_p = audio_url(audio)
				instance.audiofile=audio_p

			text=audio_to_text(audio)
			audio_name , text_name , image_name = filename(audio)
			text_path , text_p = text_url(text, text_name)
			image_path , image_p = image_url(text,image_name)
			instance.textfile=text_p
			instance.imagefile=image_p
			instance.content=text
			instance.save()
			audio=None
			data = {}
			data['text']=text
			data['image']=image_name
			# data['image']	=instance.imagefile.url
			json_data = json.dumps(data)
			return HttpResponse(json_data, content_type="application/json")
		else:
			form=UploadAudio()
			context={
			"form":form,
			}
			return render(request,'aud2gest/home.html',context)
	else:
		return redirect("../login")

def index(request):
	if request.user.is_authenticated:
		return render(request,'aud2gest/index.html',{})
	else:
		return redirect('../login')


@csrf_exempt
def ajax(request):
	if request.user.is_authenticated:
		filename="voice_"+str(randint(1000,9999))
		request.session["filename"]=filename+".wav"

		file_obj = request.FILES['audio'].read()
		print(type(file_obj))
		with default_storage.open('Z:/BTP/knk/media/aud2gest/audioFiles/'+filename+".bin", 'wb+') as destination:
			destination.write(file_obj)
			src = "Z:/BTP/knk/media/aud2gest/audioFiles/"+filename+".bin"
			dst = "Z:/BTP/knk/media/aud2gest/audioFiles/"+filename+".wav"
			sound = AudioSegment.from_file(src)
			sound.export(dst, format="wav")
			print('File Stored @ audio')
		os.remove(src) # to delete the .bin file
		return redirect("../home")
	else:
		return redirect('../login')

def about_project(request):
	return render(request, 'aud2gest/about_project.html', {})

def about_team(request):
	return render(request, 'aud2gest/about_team.html', {})

def instruction(request):
	return render(request, 'aud2gest/instructions.html', {})

