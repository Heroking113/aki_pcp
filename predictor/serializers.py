from rest_framework import serializers

from .models import Statistics, FileData


class StatisticsSerializer(serializers.ModelSerializer):

    class Meta:
        model = Statistics
        fields = '__all__'

    def to_representation(self, instance):
        """
        """
        data = super().to_representation(instance)
        data.update(gender=instance.get_gender_display())
        data.update(me_ve=instance.get_me_ve_display())
        data.update(proba="%.2f%%" % (float(instance.proba) * 100))
        return data


class FileDataSerializer(serializers.ModelSerializer):

    class Meta:
        model = FileData
        fields = '__all__'
