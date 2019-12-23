from django.shortcuts import render, redirect
from .forms import UserRegisterationForm
from django.contrib.auth import login, authenticate, logout
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User

# def register(request):
# 	if request.method == 'POST':
# 		# print(request.POST)
# 		mutable= request.POST._mutable
# 		request.POST._mutable=True
# 		request.POST['password1']=['qwerty123abc']
# 		request.POST['password2']=['qwerty123abc']
# 		request.POST._mutable=mutable
# 		print(request.POST)
# 		form = UserCreationForm(request.POST)
# 		if form.is_valid():
# 			form.save()
# 			username = form.cleaned_data.get('username')
# 			raw_password = form.cleaned_data.get('password1')
# 			print(username,raw_password)
# 			user = authenticate(username=username, password=raw_password)
# 			login(request, user)
# 			return redirect('../home/')
# 	form = UserCreationForm()
# 	return render(request, 'user/index.html', {'form': form})

def register(request):
	if not request.user.is_authenticated:
		form = UserRegisterationForm(request.POST or None)
		if request.method=='POST':
			print(request.POST)
			if form.is_valid():
				form.save()
				return redirect('../index/')
		context={
		"form":form,
		}
		return render(request,"user/register.html",context)
	else:
		return redirect('../index/')


def login_user(request):
	if not request.user.is_authenticated:
		if request.method=='POST':
				username= request.POST['username']
				password='qwerty123abc'
				if User.objects.filter(username=username).exists():
					user = User.objects.get(username=username)
				else:
					user =None
				if user is not None:
					login(request,user)
					return redirect('../index/')
		return render(request,'user/login.html',{})
	else:
		return redirect('../index/')


def logout_user(request):
	if request.user.is_authenticated:
		logout(request)
	return redirect('../login')

