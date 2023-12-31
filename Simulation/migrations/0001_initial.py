# Generated by Django 3.2 on 2023-06-06 07:35

from django.db import migrations, models
import django.db.models.deletion
import tinymce.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Category',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Name', models.CharField(max_length=40)),
            ],
        ),
        migrations.CreateModel(
            name='Modul',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Name', models.CharField(max_length=40)),
            ],
        ),
        migrations.CreateModel(
            name='Information',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Title', models.CharField(max_length=50)),
                ('Description', models.CharField(max_length=50)),
                ('Image', models.ImageField(blank=True, null=True, upload_to='')),
                ('Content', tinymce.models.HTMLField(blank=True, null=True)),
                ('Created', models.DateTimeField(auto_now_add=True)),
                ('Updated', models.DateTimeField(auto_now=True)),
                ('Category', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='Simulation.category')),
                ('Modul', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='Simulation.modul')),
            ],
            options={
                'ordering': ('Modul', 'Title'),
            },
        ),
    ]
