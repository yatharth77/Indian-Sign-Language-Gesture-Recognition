from django.db import models
from django.contrib.auth.models import User

class user_profile(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE)
	Email1 = models.EmailField(null=True)
	Email2 = models.EmailField(null=True)
	Email3 = models.EmailField(null=True)
	Email4 = models.EmailField(null=True)
	Email5 = models.EmailField(null=True)
	
	def __str__(self):
		return str(self.user)

		