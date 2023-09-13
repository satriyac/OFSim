from django.db import models
from tinymce.models import HTMLField
# Create your models here.
# Type : Artikel, Materi, Gambar/Ilustrasi
class Category(models.Model):
    Name = models.CharField(max_length=40, null=False, blank=False)
    def __str__(self):
        return self.Name
# Modul : Semua judul modul di TelcoLab
class Modul(models.Model):
    Name = models.CharField(max_length=40, null=False, blank=False)
    def __str__(self):
        return self.Name
# Information : Database inti
class Information(models.Model):
    Category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True)
    Modul = models.ForeignKey(Modul, on_delete=models.SET_NULL, null=True, blank=True)
    Title = models.CharField(max_length=50)
    Description = models.CharField(max_length=50)
    Image = models.ImageField(null=True, blank=True)
    Content = HTMLField(null=True, blank=True)
    Created = models.DateTimeField(auto_now_add= True)
    Updated = models.DateTimeField(auto_now = True)
    class Meta:
        ordering = ('Modul','Title')
    def __str__(self):
        return self.Title