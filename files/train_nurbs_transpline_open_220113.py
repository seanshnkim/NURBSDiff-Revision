import json
import logging
import os

import sys
import time
from torch.utils.tensorboard import SummaryWriter

root="/home/cloudest/"
platform=root+"parsenet-codebase/"
nurbsDiff=root+"NURBSDiff/"
config_path = "./config.yaml"

sys.path.append(platform)
sys.path.append(nurbsDiff)

from load_config import Config
from NURBSDiff.surf_eval import SurfEval
from NurbsNet_220107 import nurbs_world
from pytorch3d.loss import chamfer_distance

from shutil import copyfile
import shutil
import numpy as np
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tqdm import tqdm
from src.dataset import DataSetControlPointsPoisson
from src.dataset import generator_iter
from src.loss import (
    control_points_permute_reg_loss,
)
from src.loss import laplacian_loss, laplacian_on_samples_loss
from src.loss import (
    uniform_knot_bspline,
    spline_reconstruction_loss_one_sided,
)
from src.model import DGCNNControlPoints
from src.utils import rescale_input_outputs, haursdorff_distance

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

np.set_printoptions(precision=4)
config = Config(config_path)
userspace = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
logger.addHandler(handler)

num_samples_u = 40
num_samples_v = 40
model_name = 'new_model'
Transformer = nurbs_world(config)

nurbsA = SurfEval(config.grid_size, config.grid_size, dimension=3, p=3, q=3,
                    out_dim_u=num_samples_u, out_dim_v=num_samples_v, method='tc', dvc='cuda'
        )
nurbsB = SurfEval(10, 10, dimension=3, p=3, q=3,
                    out_dim_u=num_samples_u, out_dim_v=num_samples_v, method='tc', dvc='cuda'
        )
# 사용 가능한 GPU가 여러 개라면, 여러 개의 GPU를 활용, 데이터를 병렬 처리(DataParallel)하여 속도를 높인다.
if torch.cuda.device_count() > 1:
    Transformer = torch.nn.DataParallel(Transformer)
Transformer.cuda()
nurbsA.cuda()
nurbsB.cuda()

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

print("="*158)
print("Training Model: {}".format(model_name))
print("Training Configure information: \"{}".format(config_path))
logger.info("Trainable Parameters: {}".format(get_n_params(Transformer)))
print("="*158)

'''DATA LOADING'''
align_canonical = True
anisotropic = True
if_augment = True

dataset = DataSetControlPointsPoisson(
    platform + config.dataset_path,
    config.batch_size,
    splits=split_dict,
    size_v=20,
    size_u=20)

get_train_data = dataset.load_train_data(
    if_regular_points=True, align_canonical=align_canonical, anisotropic=anisotropic, if_augment=if_augment
)

get_val_data = dataset.load_val_data(
    if_regular_points=True, align_canonical=align_canonical, anisotropic=anisotropic
)

loader = generator_iter(get_train_data, int(1e10))
get_train_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)

loader = generator_iter(get_val_data, int(1e10))
get_val_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)
logger.info('Data Loading Complete')

optimizerA = optim.Adam(list(Transformer.parameters()) + list(nurbsA.parameters()), lr=config.lr)
schedulerA = ReduceLROnPlateau(
    optimizerA, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-5
)
optimizerB = optim.Adam(list(Transformer.parameters()) + list(nurbsB.parameters()), lr=config.lr)
schedulerB = ReduceLROnPlateau(
    optimizerB, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-5
)

'''MODEL LOADING'''
try:
    checkpoint = torch.load('logs/trained_models/'+pretrain_model_path)
    start_epoch = checkpoint['epoch']
    Transformer.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Pretrain model has been successfully loaded')
except:
    logger.info('No existing model, starting training from scratch...')
    start_epoch = 0



'''TESTING(22.01.10)'''
# torch.cuda.empty_cache()
# optimizer.zero_grad()
# points_, parameters, control_points, scales, _ = next(get_train_data)[0]
# control_points = Variable(
#     torch.from_numpy(control_points.astype(np.float32))
# ).cuda()
# print("outputMasked.size() : ", outputMasked.size())

# cp = torch.tanh(feats[..., : num_cpA * 3])
# w = torch.sigmoid(feats[..., num_cpA * 3 : num_cpA * 4]) + 1e-7
# w = w.view(-1, config.grid_size, config.grid_size, 1)

# ch = 32
# for i in range(numBlock - 1): # numBlock = 5 
#     ch += 32 * (2 ** (i+1)) # ch = 992
    
# modules_cp_w = []
# modules_cp_w.append(nn.Linear(ch, cp_ch)) # ch = 992, cp_ch = 800
# modules_cp_w.append(nn.LeakyReLU(negative_slope=0.2))
# for i in range(num_cp_layer-2):
#     modules_cp_w.append(nn.Linear(cp_ch, cp_ch)) 
#     modules_cp_w.append(nn.LeakyReLU(negative_slope=0.2))
        
# fc_cp_w = nn.Sequential(*modules_cp_w)
# last_mlp = nn.Linear(cp_ch, num_cp*4) 

# if torch.cuda.device_count() > 1:
#     fc_cp_w, last_mlp = nn.DataParallel(fc_cp_w), nn.DataParallel(last_mlp)
# fc_cp_w.cuda()
# last_mlp.cuda()

# num_cpA = (config.grid_size**2) # 400
# num_cpB = (grid_sizeB**2)       # 100

# output = last_mlp(fc_cp_w(featMax)) # output : (36, 2000)

# batch_size = output.size()[0] # 36

# cpA = outputMasked[..., : self.num_cpA*3] # 0~1200 
# cpB = outputMasked[..., self.num_cpA*4 : self.num_cpA*4 + self.num_cpB*3] # 1600 ~ 1900
# cpAB = torch.cat((cpA, cpB), dim = 1 )
# cp = torch.tanh(cpAB)

# wA = outputMasked[..., self.num_cpA*3 : self.num_cpA*4] # 1200 ~ 1600  
# wB = outputMasked[..., self.num_cpA*4 + self.num_cpB : ] # 1900 ~ 2000
# wAB = torch.cat((wA, wB), dim = 1)
# w = torch.sigmoid(wAB) + 1e-7
# w = w.view(-1, config.grid_size + grid_sizeB, config.grid_size + grid_sizeB, 1)

                    
'''TRAINING'''
k = 0

best_e = 0
best_cd = 0
best_hd = 0
best_lap = 0

prev_test_cd = 1e8
check_time = time.time()

prev = 0
for e in range(start_epoch, config.epochs + 1): # start_epoch, config.epochs + 1
#for e in range(1):
    train_it = 0
    AProb = 0
    BProb = 0
    train_hd = []
    train_cd = []
    train_lap = []
    Transformer.train()

    print("=" * 158)
    pbar = tqdm(range(config.num_train//config.batch_size), smoothing=0.9)
    for train_b_id in pbar:
    #for _ in range(1):
        
        ## CUDA, optimizer, train data 받아오는 과정
        torch.cuda.empty_cache()
        optimizerA.zero_grad()
        optimizerB.zero_grad()
        points_, parameters, control_points, scales, _ = next(get_train_data)[0]
        control_points = Variable(
            torch.from_numpy(control_points.astype(np.float32))
        ).cuda()
        
        ## random하게 points를 구해서(sampling?) Transforemr라는 이름의 네트워크에 input으로 넣고 
        ## 그 결과값인 output, weight을 얻는다.
        points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
        l = np.arange(config.num_point)
        np.random.shuffle(l)
        l = l[0:config.num_point]

        output, weight = Transformer(points[:, l, :])
#         print("\noutput.size() : ", output.size())
#         print("\nweight.size() : ", weight.size())
        AProb += Transformer.maskMatrix[0][0].item()
        BProb += Transformer.maskMatrix[0][1].item()
        if (train_it + 9) % 36 == 0:
            print("A probility / B Probability : ", AProb, BProb)

        points = points.permute(0, 2, 1)
        if anisotropic:
            scales, output, points, control_points = rescale_input_outputs(
                scales, output.view(config.batch_size, -1, 3), points, control_points,
                config.batch_size)
        
        ## loss를 구하기 위해 네트워크의 결과값인 output을 NURBS에 넣어 reconstructed points를 구한다
        # output = output.view(-1, config.grid_size, config.grid_size, 3)
        output = output.view(-1, weight.size()[1], weight.size()[2], 3)
        output = torch.cat((output, weight), -1)
        
        if weight.size()[1] == config.grid_size:
            output = nurbsA(output)
        elif weight.size()[1] == 10:
            output = nurbsB(output)
            
        recon_prediction = output.view(-1, num_samples_u * num_samples_v, 3)
        
        ## loss를 구하는 과정
        cd = chamfer_distance(recon_prediction, points.permute(0, 2, 1))
        #print("\ncd : ", cd)
        hd = haursdorff_distance(recon_prediction, points.permute(0, 2, 1))
        #print("\nhd : ", hd)
        laplac_loss = laplacian_on_samples_loss(
            recon_prediction.reshape((config.batch_size, num_samples_u, num_samples_v, 3)) )
        #print("\nlaplac_loss : ", laplac_loss)

        loss = cd[0] + hd + 0.1 * laplac_loss
        if e > 10 and (loss - prev > 0.05):
            logger.info("[error] ctrl_pts: {}".format(ctrl))
            logger.info("[error] weight: {}".format(weight))
            logger.info("[error] knots_u: {}".format(knots_u))
            logger.info("[error] knots_v: {}".format(knots_v))
        prev = loss
        loss.backward(retain_graph=True)
        train_cd.append(cd[0].data.cpu().numpy())
        train_hd.append(hd.data.cpu().numpy())
        train_lap.append(laplac_loss.data.cpu().numpy())
        if weight.size()[1] == config.grid_size:
            optimizerA.step()
        elif weight.size()[1] == 10:
            optimizerB.step()
        
        pbar.set_description_str(desc="[Train] Epoch {}/{}, loss: {:.7f}".format(e, config.epochs, loss.item()), refresh=True)
        # print("[Train] Epoch {}/{}, loss: {:.7f}".format(e, config.epochs, loss.item()))
        k += 1
        
        train_it += 1

    ## validation 과정 준비
    test_cd = []
    test_hd = []
    test_lap = []
    Transformer.eval()
    
    # validation 과정
    pbar = tqdm(range(config.num_val//config.batch_size), smoothing=0.9)
    for val_b_id in pbar:
        torch.cuda.empty_cache()
        points_, parameters, control_points, scales, _ = next(get_val_data)[0]
        control_points = Variable(
            torch.from_numpy(control_points.astype(np.float32))
        ).cuda()

        points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
        with torch.no_grad():
            output, weight = Transformer(points[:, 0:config.num_point, :])
            points = points.permute(0, 2, 1)
            if anisotropic:
                scales, output, points, control_points = rescale_input_outputs(scales, output.view(config.batch_size, -1, 3), points, control_points,
                                                                               config.batch_size)

        output = output.view(-1, weight.size()[1], weight.size()[2], 3)
        output = torch.cat((output, weight), -1)
        if weight.size()[1] == config.grid_size:
            output = nurbsA(output)
        elif weight.size()[1] == 10:
            output = nurbsB(output)
        recon_prediction = output.view(-1, num_samples_u * num_samples_v, 3)

        cd = chamfer_distance(recon_prediction, points.permute(0, 2, 1))
        hd = haursdorff_distance(recon_prediction, points.permute(0, 2, 1))
        laplac_loss = laplacian_on_samples_loss(
            recon_prediction.reshape((config.batch_size, num_samples_u, num_samples_v, 3))
        )

        loss = cd[0] + hd + 0.1*laplac_loss
        test_cd.append(cd[0].data.cpu().numpy())
        test_hd.append(hd.data.cpu().numpy())
        test_lap.append(laplac_loss.data.cpu().numpy())
        pbar.set_description_str(desc="[Valid] loss: {:.7f}".format(loss.item()), refresh=True)
        # print("[Valid] loss: {:.7f}".format(loss.item()) )

    logger.info(
        "[Result] CD:{:.7f}/{:.7f}, HD:{:.7f}/{:.7f}, LAP:{:.7f}/{:.7f}".format(
            np.mean(train_cd),
            np.mean(test_cd),
            np.mean(train_hd),
            np.mean(test_hd),
            np.mean(train_lap),
            np.mean(test_lap),
        )
    )

    scheduler.step(np.mean(test_cd))
    if prev_test_cd > np.mean(test_cd):
        prev_test_cd = np.mean(test_cd)
        prev_test_hd = np.mean(test_hd)
        prev_test_lap = np.mean(test_lap)

        torch.save({
                'epoch': e,
                'model_state_dict': Transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "saved_models/{}.pth".format(model_name),
        )

        best_e = e
        best_cd = prev_test_cd
        best_hd = prev_test_hd
        best_lap = prev_test_lap

    if e != 0:
        logger.info("[Recorded] {} Epoch, Best CD: {:.7f} with HD:{:.7f}, LAP:{:.7f}".format(best_e, best_cd, best_hd, best_lap))
