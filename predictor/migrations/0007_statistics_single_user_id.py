# Generated by Django 2.2 on 2021-04-30 15:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0006_auto_20210430_1033'),
    ]

    operations = [
        migrations.AddField(
            model_name='statistics',
            name='single_user_id',
            field=models.CharField(default='', max_length=32, verbose_name='单一用户的ID'),
        ),
    ]