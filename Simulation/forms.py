from django import forms

class ParameterForm(forms.Form):
    input_text = forms.CharField(widget=forms.Textarea(attrs={'rows': 4, 'cols': 50}))
