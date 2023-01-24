from django import forms

class FotoURLForm(forms.Form):
    foto_url = forms.URLField()