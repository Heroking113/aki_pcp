# Generated by Django 2.2 on 2021-04-30 10:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0004_auto_20210430_0954'),
    ]

    operations = [
        migrations.AlterField(
            model_name='statistics',
            name='dead_label',
            field=models.BooleanField(default=False, verbose_name='是否死亡'),
        ),
        migrations.AlterField(
            model_name='statistics',
            name='gender',
            field=models.CharField(choices=[('1', '男性'), ('0', '女性')], default='-1', max_length=4, verbose_name='用户性别'),
        ),
        migrations.AlterField(
            model_name='statistics',
            name='me_ve',
            field=models.BooleanField(default=True, verbose_name='是否机械通风'),
        ),
        migrations.AlterField(
            model_name='statistics',
            name='user_id',
            field=models.CharField(max_length=8, verbose_name='用户索引(第一列的数据)'),
        ),
    ]
