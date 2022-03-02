from django.db import models

# Create your models here.
class Statistics(models.Model):
    GENDER = (
        ('1', '男性'),
        ('0', '女性')
    )

    ME_VE = (
        ('1', '通风'),
        ('0', '未通风')
    )

    file_id = models.IntegerField(verbose_name='文件ID', default=0)
    patient_id = models.CharField(verbose_name='单一用户的ID', max_length=32, default='0')
    user_id = models.CharField(verbose_name='用户索引(第一列的数据)', max_length=8)
    dead_label = models.BooleanField(verbose_name='是否死亡', default=False)
    gender = models.CharField(verbose_name='用户性别', max_length=4, choices=GENDER, default='-1')
    me_ve = models.CharField(verbose_name='是否机械通风', choices=ME_VE, max_length=64, default='0')
    proba = models.CharField(verbose_name='死亡概率', max_length=8, default='0') # char
    treat_advice = models.TextField(verbose_name='治疗建议', default='')
    nurse_advice = models.TextField(verbose_name='护理建议', default='')
    create_time = models.DateTimeField(verbose_name='创建的时间', auto_now_add=True)


    class Meta:
        db_table = 'statistics'
        verbose_name = '院内死亡预测统计数据'
        verbose_name_plural = verbose_name


class FileData(models.Model):
    input_data_filename = models.FileField(upload_to='input_data/', verbose_name='上传文件的路由')
    img_url = models.TextField(verbose_name='结果图片路径', default='')
    create_time = models.DateTimeField(verbose_name='创建的时间', auto_now_add=True)

    class Meta:
        db_table = 'file_data'
        verbose_name = '文件数据'
        verbose_name_plural = verbose_name
