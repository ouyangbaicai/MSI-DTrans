import os
import sys
import glob
import time

import cv2
import torch

from tqdm import tqdm
from torch import einsum
from Nets.MY_Network_gy import Network
from Utilities import Consistency
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU
from thop import profile, clever_format

# Modified parameters
j = 0
network_name = 'filename'
DATAPATH = 'your dir'
total_train = len(os.listdir(DATAPATH))

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class Fusion:
    def __init__(self,
                 modelpath,
                 network_name,
                 # modelpath='RunTimeData/2023-10-30 16.28.57/model31.ckpt',
                 dataroot='your dir',
                 dataset_name='Lytro',
                 threshold=0.001,
                 window_size=5,
                 ):
        self.DEVICE = GPUorCPU().DEVICE
        self.network_name = network_name
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.window_size = window_size
        self.window = torch.ones([1, 1, self.window_size, self.window_size], dtype=torch.float).to(self.DEVICE)

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME != None:
            self.SAVEPATH = '/' + self.network_name + self.DATASET_NAME + str(j)
            # self.SAVEPATH = '/' + self.DATASET_NAME
            self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
            MODEL = self.LoadWeights(self.MODELPATH)
            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD, self.network_name)
        else:
            print("Test Dataset required!")
            pass

    def LoadWeights(self, modelpath):
        model = Network().to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6))) # round()四舍五入 数million进制转换 保留6位
        flops, params = profile(model, inputs=(torch.rand(1, 3, 520, 520).cuda(), torch.rand(1, 3, 520, 520).cuda()))
        # flops, params = profile(model, inputs=(torch.rand(1, 3, 520, 520), torch.rand(1, 3, 520, 520)))
        flops, params = clever_format([flops, params], "%.5f")
        print('flops: {}, params: {}\n'.format(flops, params))
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        if threshold != 0:
            Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold, network_name):
        if not os.path.exists('./Results/' + savepath):
            os.makedirs('./Results/' + savepath, exist_ok=True)
        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data,
                                 batch_size=1,
                                 shuffle=False, )
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
        cnt = 1
        running_time = []
        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                start_time = time.time()

                D = model(A, B)
                D = torch.where(D > 0.5, 1., 0.)
                # 决策图生成 如果满足条件，返回1.0，否则返回0.0。
                D = self.ConsisVerif(D, threshold)
                D = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()
                A = cv2.imread(eval_list_A[cnt - 1])
                B = cv2.imread(eval_list_B[cnt - 1])
                IniF = A * D + B * (1 - D)
                D = D * 255
                # D = cv2.filtered_image = cv2.bilateralFilter(D, 15, 75, 75)
                # D = cv2.GaussianBlur(D, (5, 5), 0)
                cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '-dm.png', D)
                # cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '-dm1.png', D1)
                cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '.png', IniF)
                cnt += 1
                running_time.append(time.time() - start_time)
        running_time_total = 0
        for i in range(len(running_time)):
            print("process_time: {} s".format(running_time[i]))
            if i != 0:
                running_time_total += running_time[i]
        # print("\navg_process_time: {} s".format(running_time_total / (len(running_time) - 1)))
        print("\nResults are saved in: " + "./Results" + savepath)


if __name__ == '__main__':
    for i in range(total_train):
        datapath = DATAPATH + '/model' + str(i+1) +'.ckpt'
        f = Fusion(datapath, network_name)
        j += 1
        f()
