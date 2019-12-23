from django.db import models
# from .settings import MEDIA_URL
# Create your models here.
class AudioDb(models.Model):
	audiofile=models.FileField(upload_to='aud2gest/audioFiles/',blank=True, null=True)
	textfile=models.FileField(upload_to='aud2gest/textFiles/',blank=True,null=True)
	imagefile=models.FileField(upload_to='aud2gest/imageFiles/', blank=True,null=True)
	content=models.TextField(blank=True,null=True)
	# voice_record = models.FileField()