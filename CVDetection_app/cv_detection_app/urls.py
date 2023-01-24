from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('get_detection/', views.get_detection, name='get_detection'),
]