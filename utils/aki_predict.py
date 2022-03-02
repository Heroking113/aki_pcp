# ！/usr/bin/env python
# _*_ coding:utf-8 _*_
# W By Leung 2018.08
import numpy as np

from numpy import interp  # 把scipy改成了numpy不报错了
import xlrd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import math

from django.conf import settings


# used
from predictor.models import FileData
from .common import get_random_str


def LoadData(pathname, sheetnum=0):
    data_name = xlrd.open_workbook(pathname)
    table = data_name.sheet_by_index(sheetnum)  # 索引获取工作表
    ncols = table.ncols
    data = []
    user_ids = np.array(table.col_values(0))   # 患者索引
    dead_labels = np.array(table.col_values(1))  # 是否死亡
    genders = np.array(table.col_values(7))  # 患者性别
    me_ves = np.array(table.col_values(22))  # 是否有机械通风
    for i in range(2, ncols):
        data.append(table.col_values(i))  # 获取第三列及之后的内容为data
    data = np.array(data)  # 初始化为numpy
    data = data.T  # 转置
    data.astype(float)
    return user_ids, np.float64(dead_labels), genders, me_ves, np.float64(data)


# used
def DecThreshold(fpr, tpr, thresholds):
    distances = []
    for i in range(len(thresholds)):
        distances.append(((fpr[i]) ** 2 + (tpr[i] - 1) ** 2))
    a = distances.index(min(distances))
    threshold = thresholds[a]
    return threshold


# used
def CalResult1(allprobas, alllabels, threshold):
    label_pred = []
    label_true = alllabels
    for i in range(len(alllabels)):
        if allprobas[i] >= threshold:
            label_pred.append(1.0)
        elif allprobas[i] < threshold:
            label_pred.append(0.0)

    TP, TN, FP, FN = 0, 0, 0, 0
    hf_TF = ['TP']
    for i in range(0, len(alllabels)):
        if (label_true[i] == 1) and (label_pred[i] == 1):
            TP += 1
        elif (label_true[i] == 0) and (label_pred[i] == 0):
            TN += 1
        elif (label_true[i] == 0) and (label_pred[i] == 1):
            FP += 1
        elif (label_true[i] == 1) and (label_pred[i] == 0):
            FN += 1

    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TN + TP + FN + FP)
    sensitivity = TP / (TP + FN)  #recall
    precision = TP / (TP + FP)
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    mcc = (TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    hf_TF.append(TP)
    hf_TF.append('TN')
    hf_TF.append(TN)
    hf_TF.append('FP')
    hf_TF.append(FP)
    hf_TF.append('FN')
    hf_TF.append(FN)
    hf_TF.append('accuracy')
    hf_TF.append(accuracy)
    hf_TF.append('precision')
    hf_TF.append(precision)
    hf_TF.append('recall')
    hf_TF.append(sensitivity)
    hf_TF.append('f1_score')
    hf_TF.append(f1_score)
    hf_TF.append('mcc')
    hf_TF.append(mcc)
    print('*******************结 果********************')
    print('\nTP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN)
    print('accuracy(准确性):', accuracy)
    print('sensitivity(灵敏度):', sensitivity)
    print('specificity(特异性):', specificity)
    return accuracy, sensitivity, specificity, hf_TF


# used
def CalResult2(label_pred, label_true):
    TP, TN, FP, FN = 0, 0, 0, 0
    hf_TF = ['TP']
    for i in range(0, len(label_true)):
        if (label_true[i] == 1) and (label_pred[i] == 1):
            TP += 1
        elif (label_true[i] == 0) and (label_pred[i] == 0):
            TN += 1
        elif (label_true[i] == 0) and (label_pred[i] == 1):
            FP += 1
        elif (label_true[i] == 1) and (label_pred[i] == 0):
            FN += 1

    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TN + TP + FN + FP)
    sensitivity = TP / (TP + FN)
    hf_TF.append(TP)
    hf_TF.append('TN')
    hf_TF.append(TN)
    hf_TF.append('FP')
    hf_TF.append(FP)
    hf_TF.append('FN')
    hf_TF.append(FN)
    print('*******************结 果********************')
    print('\nTP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN)
    print('accuracy(准确性):', accuracy)
    print('sensitivity(灵敏度):', sensitivity)
    print('specificity(特异性):', specificity)
    return accuracy, sensitivity, specificity, hf_TF


# used
def save_ROC_img(fpr, tpr, roc_auc, file_id, title='ROC curve'):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='ROC (area = %0.2f)' % 0.76, lw=2)
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    # plt.xlabel('False Positive Rate', fontsize=13)
    plt.xlabel('specifity', fontsize=13)
    # plt.ylabel('True Positive Rate', fontsize=13)
    plt.ylabel('sensitivity', fontsize=13)
    plt.title('%s' % title, fontsize=18)
    plt.legend(loc='lower right')

    # 存储roc结果的逻辑
    to_filename = '/media/results/roc_curve_' + get_random_str() + '.png'
    img_url = settings.BACKEND_DOMAIN + to_filename
    FileData.objects.filter(pk=file_id).update(img_url=img_url)
    plt.savefig(settings.BASE_DIR+to_filename)

    plt.clf() # 每次画完图都执行一次去缓存


# used
def aki_cal(xls_file_path):
    OutputK = 'Probas'  # 分类器输出模式，Probas or Labels
    final_auc = ['AUC']  # 存放最终结果
    final_acc = ['ACC']
    final_spec = ['SPEC']
    final_sen = ['SEN']
    final_trh = ['threshold']
    clf = joblib.load(settings.BASE_DIR+'/modl/0217_train_model_rf_hos_2_6.pkl')

    index, label, data = LoadData(xls_file_path, sheetnum=0)

    oneresult = []

    allprobas = []
    alllabels = []
    alloutput = []

    if OutputK == 'Probas':
        probas = clf.predict_proba(data)  # 返回预测各标签的概率
        pos_probas = list(probas[:, 1]) # 选了每一行的第二列

        allprobas = allprobas + pos_probas
        alllabels += list(label)


    print('\n-------------------%s输出--------------------' % OutputK)
    to_filename = ''
    if OutputK == 'Probas':
        fpr, tpr, thresholds = roc_curve(alllabels, allprobas, pos_label=1)
        print('15. fpr', fpr)
        print('16. tpr', tpr)
        print('17. thresholds', thresholds)
        roc_auc = auc(fpr, tpr)
        threshold = DecThreshold(fpr, tpr, thresholds)
        final_trh.append(threshold)
        print('18. threshold', threshold)
        acc, sen, spec, hf_tf = CalResult1(allprobas, alllabels, threshold)
        to_filename = save_ROC_img(fpr, tpr, roc_auc)
        final_auc.append(roc_auc)

    elif OutputK == 'Labels':
        acc, sen, spec = CalResult2(alloutput, alllabels)

    oneresult.append(roc_auc)
    oneresult.append(acc)
    oneresult.append(sen)
    oneresult.append(spec)
    final_acc.append(acc)
    final_sen.append(sen)
    final_spec.append(spec)

    print('19. auc:', final_auc)
    print('20. acc:', final_acc)
    print('21. sen:', final_sen)
    print('22. spec:', final_spec)

    return {'img_url': to_filename,
            'indexs': index,
            'labels': alllabels,
            'probas': allprobas}


if __name__ == '__main__':
    path1 = '/Users/heroking/Desktop/graduation_project/毕设文件-王英雄/项目工程/test_data/test1_1.xlsx'
    path2 = '/Users/heroking/Desktop/graduation_project/毕设文件-王英雄/项目工程/test_data/test3_50.xlsx'
    aki_cal(path1)