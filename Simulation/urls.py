from django.urls import path
from . import views

urlpatterns = [
    path('ofdmsim', views.ofdmsim, name='ofdmsim' ),
    path('', views.ofdmTransceiver, name='ofdmTransmitter'),
]