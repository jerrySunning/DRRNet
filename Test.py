import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2

from model.DRRNet_PVT import Network
from utils.data_val import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')  #
parser.add_argument('--pth_path', type=str,
                    default=r'Net_epoch_best_DRRNet.pth')
parser.add_argument('--test_dataset_img_path', type=str, default='D:\\allCode\\allDatasets\\COD_Test\\')
parser.add_argument('--test_dataset_gt_path', type=str, default='D:\\allCode\\allDatasets\\COD_Test\\')

opt = parser.parse_args()


for _data_name in ['COD10K', 'NC4K','CAMO']:
    print("-------------------------TEST" + _data_name + "------------------------")
    save_path = "testoutput" + _data_name + "\\"
    os.makedirs(save_path, exist_ok=True)

    model = Network(channels=128)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.pth_path).items()})
    model.cuda()
    model.eval()

    image_root = opt.test_dataset_img_path + _data_name + "\\Images\\"
    gt_root = opt.test_dataset_gt_path + _data_name + "\\GT\\"
    testsize = opt.testsize
    test_loader = test_dataset(image_root, gt_root, testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        name = name.split('\\')[-1]
        print('> {} - {} - {}'.format(_data_name, i + 1, name))

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        result = model(image)

        res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)




