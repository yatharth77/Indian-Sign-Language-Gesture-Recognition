from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import user_profile

class UserRegisterationForm(UserCreationForm):
	# username = forms.CharField(label="Your Username")
	# password1 = forms.CharField(label="Your Password")
	# password2 = forms.CharField(label="Repeat Your Password")
	Email1 = forms.EmailField(label = "Email Address 1")
	Email2 = forms.EmailField(label = "Email Address 2")
	Email3 = forms.EmailField(label = "Email Address 3")
	Email4 = forms.EmailField(label = "Email Address 4")
	Email5 = forms.EmailField(label = "Email Address 5")
 
	class Meta:
		model = User
		fields = ("username","password1", "password2","Email1","Email2","Email3","Email4","Email5")

	def save(self, commit=True):
		user = super(UserRegisterationForm, self).save(commit=False)
		
		if commit:
			user.save()

		profile = user_profile(user=user, Email1=self.cleaned_data['Email1'], Email2=self.cleaned_data['Email2'],Email3=self.cleaned_data['Email3'],Email4=self.cleaned_data['Email4'],Email5=self.cleaned_data['Email5'])
		profile.save()

		return user,user_profile