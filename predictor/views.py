import shutil

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import FileData, Statistics
from .serializers import StatisticsSerializer
from .tasks import aki_cal, aki_cal_sp


class AkiView(APIView):

    def get(self, request, *args, **kwargs):
        patient_id = int(request.query_params.get('patient_id', -1))
        file_id = int(request.query_params.get('file_id', -1))

        statistic_query = Statistics.objects.filter(file_id=file_id)
        if patient_id:
            statistic_query = Statistics.objects.filter(patient_id=patient_id)
        static_serializer = StatisticsSerializer(statistic_query, many=True)
        patients = [dict(item) for item in static_serializer.data]

        img_url = settings.IMG_PATH_PREFIX + FileData.objects.first().img_url
        ret_data = {'patients': patients, 'img_url': img_url}

        return Response(ret_data)


    def post(self, request, *args, **kwargs):
        file = request.data.get('file', '')
        li_data = request.data.get('li_data')
        patient_id = request.data.get('patient_id', '')
        # aki_cal.delay('123', '456')
        # return Response()
        if file:
            shutil.rmtree(settings.BASE_DIR+'/media/input_data/')
            file_data = FileData.objects.create(input_data_filename=file)
            filepath = settings.BASE_DIR + file_data.input_data_filename.url
            file_id = file_data.id
            aki_cal.delay(filepath, file_id)
            return Response({'errmsg': 'success', 'file_id': file_id})
        else:
            aki_cal_sp.delay(li_data, patient_id)
            return Response({'errmsg': 'success'})


@api_view(http_method_names=['POST'])
def update_statistic(request):
    pk = request.data.get('id', 0)
    treat_advice = request.data.get('treat_advice', None)
    nurse_advice = request.data.get('nurse_advice', None)
    update_data = {}
    if treat_advice:
        update_data.update({'treat_advice': treat_advice})
    if nurse_advice:
        update_data.update({'nurse_advice': nurse_advice})

    Statistics.objects.filter(pk=pk).update(**update_data)
    return Response()