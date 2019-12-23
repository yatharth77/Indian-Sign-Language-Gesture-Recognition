"""knk URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
	https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
	1. Add an import:  from my_app import views
	2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
	1. Add an import:  from other_app.views import Home
	2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
	1. Import the include() function: from django.urls import include, path
	2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.conf import settings
from django.contrib import admin
from django.urls import path
from aud2gest import views as aud_view
from user import views as user_view
from gest2aud import views as gest_view

urlpatterns = [ 
	path('admin/', admin.site.urls),
	path('home/', aud_view.home, name="upload_audio"),
	path('index/', aud_view.index, name="index"),
	path('save_audio/', aud_view.ajax, name="save_audio"),
	path('about_project/', aud_view.about_project, name="about_project"),
	path('about_team/', aud_view.about_team, name="about_team"),
	path('instruction/', aud_view.instruction, name="instruction"),
	path('register/', user_view.register, name="register"),
	path('login/', user_view.login_user, name="login_user"),
	path('webcam/', gest_view.take_snaps, name="webcam"),
	path('gest_keyboard/', gest_view.gest_keyboard, name="gest_keyboard"),
	path('logout/', user_view.logout_user, name="logout"),
	path('emergency/', gest_view.emergency, name='emergency')

]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
