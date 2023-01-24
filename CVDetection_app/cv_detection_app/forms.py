from django import forms

class FotoURLForm(forms.Form):
    foto_url = forms.URLField()
    confidence = forms.FloatField(widget=forms.NumberInput(attrs={'type':'range', 'step': '0.1'}))