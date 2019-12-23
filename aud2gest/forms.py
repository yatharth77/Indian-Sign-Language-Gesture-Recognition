from django import forms
from .models import AudioDb
from django.forms import ModelForm

class UploadAudio(ModelForm):
	class Meta:
		model = AudioDb
		fields = ['audiofile']
		widgets = {
				'audiofile': forms.FileInput(attrs={'id': 'post-text'}),		
		}