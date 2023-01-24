from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .forms import FotoURLForm

from cv_detection_app import detector
import sys

def index(request):

    context = {"detection": False}

    return render(request, 'cv_detection_app/index.html', context)


def get_detection(request):

    #uris = ['http://lazysmart.ru/wp-content/uploads/2023/01/Snimok4.jpg']

    detection = False
    error_message = "Что-то не так со ссылкой"

    if request.method == 'POST':
        form = FotoURLForm(request.POST)
        if form.is_valid():
            foto_url = form.cleaned_data['foto_url'] 
            confidence = form.cleaned_data['confidence'] 
            results= detector.predict([foto_url,], confidence)
            detector.draw_results(*results)     
            detection = True
            error_message = False
        else:
            error_message = form.errors['foto_url']

    context = {"detection": detection,
               "error_message": error_message,
               "foto_url": foto_url,
               "confidence": confidence} 

    return render(request, 'cv_detection_app/index.html', context)
    