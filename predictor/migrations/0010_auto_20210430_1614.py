# Generated by Django 2.2 on 2021-04-30 16:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0009_auto_20210430_1552'),
    ]

    operations = [
        migrations.AlterField(
            model_name='statistics',
            name='me_ve',
            field=models.CharField(default='1', max_length=4, verbose_name='是否机械通风'),
        ),
    ]
