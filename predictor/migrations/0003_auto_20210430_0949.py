# Generated by Django 2.2 on 2021-04-30 09:49

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0002_auto_20210430_0857'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='statistics',
            name='ages',
        ),
        migrations.RemoveField(
            model_name='statistics',
            name='dead_labels',
        ),
        migrations.RemoveField(
            model_name='statistics',
            name='genders',
        ),
        migrations.RemoveField(
            model_name='statistics',
            name='idxs',
        ),
        migrations.RemoveField(
            model_name='statistics',
            name='los_icus',
        ),
        migrations.AddField(
            model_name='statistics',
            name='create_time',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now, verbose_name='创建的时间'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='statistics',
            name='dead_label',
            field=models.IntegerField(default=-1, verbose_name='死亡标签(第二列的数据)'),
        ),
        migrations.AddField(
            model_name='statistics',
            name='gender',
            field=models.IntegerField(default=-1, verbose_name='用户性别'),
        ),
        migrations.AddField(
            model_name='statistics',
            name='me_ves',
            field=models.IntegerField(default=-1, verbose_name='机械通风'),
        ),
        migrations.AddField(
            model_name='statistics',
            name='nurse_advice',
            field=models.TextField(default='', verbose_name='护理建议'),
        ),
        migrations.AddField(
            model_name='statistics',
            name='treat_advice',
            field=models.TextField(default='', verbose_name='治疗建议'),
        ),
        migrations.AddField(
            model_name='statistics',
            name='user_id',
            field=models.CharField(default=0, max_length=16, verbose_name='用户索引(第一列的数据)'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='statistics',
            name='probas',
            field=models.CharField(default='0', max_length=8, verbose_name='死亡概率'),
        ),
    ]
