from __future__ import absolute_import
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from utils.aki_predict import LoadData, DecThreshold, CalResult1, CalResult2, save_ROC_img  # , save_ROC_img
from celery import shared_task

from django.conf import settings

from utils.common import res_persistence


@shared_task
def aki_cal(file_or_dic, file_id):
    OutputK = 'Probas'  # 分类器输出模式，Probas or Labels
    final_trh = ['threshold']
    clf = joblib.load(settings.BASE_DIR+'/modl/0217_train_model_rf_hos_2_6.pkl')

    user_ids, dead_labels, genders, me_ves, data = LoadData(file_or_dic, sheetnum=0)

    alllabels = []
    alloutput = []
    allprobas = []

    if OutputK == 'Probas':
        probas = clf.predict_proba(data)  # 返回预测各标签的概率
        pos_probas = list(probas[:, 1]) # 选了每一行的第二列

        allprobas = pos_probas

        if isinstance(dead_labels, float):
            alllabels.append(int(dead_labels))
        else:
            alllabels += list(dead_labels)


    print('\n-------------------%s输出--------------------' % OutputK)
    if OutputK == 'Probas':
        fpr, tpr, thresholds = roc_curve(alllabels, allprobas, pos_label=1)
        roc_auc = auc(fpr, tpr)
        threshold = DecThreshold(fpr, tpr, thresholds)
        final_trh.append(threshold)
        if len(allprobas) > 1:
            acc, sen, spec, hf_tf = CalResult1(allprobas, alllabels, threshold)
            print('acc:', acc)
            print('sen:', sen)
            print('spec:', spec)
            print('hf_tf:', hf_tf)
        save_ROC_img(fpr, tpr, roc_auc, file_id)

    elif OutputK == 'Labels':
        acc, sen, spec = CalResult2(alloutput, alllabels)

    user_ids = [int(i) for i in user_ids]
    if isinstance(dead_labels, np.float64):
        dead_labels = [bool(dead_labels)]
    else:
        dead_labels = [bool(i) for i in dead_labels]
    genders = [str(int(i)) for i in genders]
    me_ves = [str(int(i)) for i in me_ves]
    allprobas = [str(item)[:6] for item in allprobas]
    res_persistence(file_id, user_ids, dead_labels, genders, me_ves, allprobas)

@shared_task
def aki_cal_sp(li_data, patient_id):
    user_ids = [li_data[0]]
    dead_labels = [False]
    genders = [li_data[6]]
    me_ves = [li_data[21]]

    clf = joblib.load(settings.BASE_DIR + '/modl/0217_train_model_rf_hos_2_6.pkl')
    cl_data = [[item] for item in li_data[1:]]
    cl_data = np.array(cl_data)  # 初始化为numpy
    cl_data = cl_data.T  # 转置
    cl_data.astype(float)

    probas = clf.predict_proba(np.float64(cl_data))  # 返回预测各标签的概率
    probas = list(probas[:, 1])

    res_persistence(file_id=0, user_ids=user_ids, dead_labels=dead_labels, genders=genders, me_ves=me_ves, probas=probas, patient_id=patient_id)
