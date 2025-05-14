import numpy as np
from test_data import test_dataset
from PIL import Image
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm
dataset_path = 'D:\\allCode\\allDatasets\\COD_Test\\'
dataset_path_pre = 'D:\\allCode\\allExperiment\\FUSIONet\\output\\'
import os
test_datasets = ['COD10K', 'CAMO', 'NC4K']
file_path = 'result.txt'



def calculate_single_metrics(sal_path, gt_path):
    sal = Image.open(sal_path)
    gt = Image.open(gt_path)

    if sal.size != gt.size:
        x, y = gt.size
        sal = sal.resize((x, y))

    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    gt[gt > 0.5] = 1
    gt[gt != 1] = 0

    res = np.array(sal)
    if res.max() == res.min():
        res = res / 255
    else:
        res = (res - res.min()) / (res.max() - res.min())

    mae_calculator = cal_mae()
    fm_calculator = cal_fm(1)
    sm_calculator = cal_sm()
    em_calculator = cal_em()
    wfm_calculator = cal_wfm()

    mae_calculator.update(res, gt)
    sm_calculator.update(res, gt)
    fm_calculator.update(res, gt)
    em_calculator.update(res, gt)
    wfm_calculator.update(res, gt)

    MAE = mae_calculator.show()
    maxf, meanf, _, _ = fm_calculator.show()
    sm_result = sm_calculator.show()
    em_result = em_calculator.show()
    wfm_result = wfm_calculator.show()

    print('MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(MAE, maxf, meanf, wfm_result,
                                                                                      sm_result, em_result))
    return MAE, maxf, meanf, wfm_result, sm_result, em_result



for dataset in test_datasets:
    sal_root = dataset_path_pre + dataset + '\\'
    gt_root = dataset_path + dataset + "\\GT\\"
    test_loader = test_dataset(sal_root, gt_root)
    mae,fm,sm,em,wfm= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm()
    for i in range(test_loader.size):
        print ('predicting for %d / %d' % ( i + 1, test_loader.size))
        sal, gt = test_loader.load_data()
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res/255
        else:
            res = (res - res.min()) / (res.max() - res.min())
        mae.update(res, gt)
        sm.update(res,gt)
        fm.update(res, gt)
        em.update(res,gt)
        wfm.update(res,gt)

    MAE = mae.show()
    maxf,meanf,_,_ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(dataset, MAE, maxf,meanf,wfm,sm,em))



    file_path = 'result.txt'
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(dataset, MAE, maxf,meanf,wfm,sm,em))
    file.close()



