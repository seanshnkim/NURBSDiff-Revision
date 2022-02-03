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
from NurbsNet_220107 import Backbone
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
# Transformer = nurbs_world(config)

nurbs = SurfEval(config.grid_size, config.grid_size, dimension=3, p=3, q=3,
                    out_dim_u=num_samples_u, out_dim_v=num_samples_v, method='tc', dvc='cuda'
        )
# 사용 가능한 GPU가 여러 개라면, 여러 개의 GPU를 활용, 데이터를 병렬 처리(DataParallel)하여 속도를 높인다.
# if torch.cuda.device_count() > 1:
#     Transformer = torch.nn.DataParallel(Transformer)
# Transformer.cuda()
nurbs.cuda()

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

# print("="*158)
# print("Training Model: {}".format(model_name))
# print("Training Configure information: \"{}".format(config_path))
# logger.info("Trainable Parameters: {}".format(get_n_params(Transformer)))
# print("="*158)

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

# optimizer = optim.Adam(list(Transformer.parameters()) + list(nurbs.parameters()), lr=config.lr)
# scheduler = ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-5
# )

# '''MODEL LOADING'''
# try:
#     checkpoint = torch.load('logs/trained_models/'+pretrain_model_path)
#     start_epoch = checkpoint['epoch']
#     Transformer.load_state_dict(checkpoint['model_state_dict'])
#     logger.info('Pretrain model has been successfully loaded')
# except:
#     logger.info('No existing model, starting training from scratch...')
#     start_epoch = 0



'''TESTING(22.01.10)'''
torch.cuda.empty_cache()
# optimizer.zero_grad()
points_, parameters, control_points, scales, _ = next(get_train_data)[0]
control_points = Variable(
    torch.from_numpy(control_points.astype(np.float32))
).cuda()

points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
l = np.arange(config.num_point)
np.random.shuffle(l)
l = l[0:config.num_point]

enc1 = Backbone(config)

if torch.cuda.device_count() > 1:
    enc1 = nn.DataParallel(enc1)
enc1.cuda()

# points[:, l, :] : 36, 1024, 3
_, xyz_and_features = enc1(points[:, l, :]) # len(xyz_and_features) = 5

# ##### case 1 #####
# feats = torch.max(xyz_and_features[0][1], dim=1).values
# print("feats.size() : ", feats.size()) # 36, 32

# for i in range(1, len(xyz_and_features)):
#     temp = torch.max(xyz_and_features[i][1], dim=1).values
#     print("temp size() : ", temp.size() )
#     feats = torch.cat((feats, temp), dim=1)
#     print("feats size() : ", feats.size() )
# print("feats size : ", feats.size() ) # 36,992

''' len(features) = 5
    len(features[0...4]) = 36
    
        len(features[0][0...35] = 1024
            len(features[0][0][0...1023] = 32
            
        len(features[1][0...35] = 256
            len(features[1][0][0...255] = 64
            
        len(features[2][0...35] = 64
            len(features[2][0][0...63] = 128
            
        len(features[3][0...35] = 16
            len(features[3][0][0...15] = 256
            
        len(features[4][0...35] = 4
            len(features[4][0][0...3] = 512 '''

features = [ elem[1] for elem in xyz_and_features ]
numBlock = len(features) # 5

featMax = torch.max(features[0], dim=1).values
for i in range(1,numBlock):
    temp = torch.max(features[i], dim=1).values
    featMax = torch.cat((featMax, temp), dim = 1) # featMax.size() = (36, 992)

#########################################################################################################################
class MLP_GumbelSoftmax(nn.Module):
    def __init__(self, nF, nC):
        super().__init__()
        self.fc1 = nn.Linear(nF, 80, bias=True)
        self.fc2 = nn.Linear(80, nC, bias=True)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + 1e-5
        # xLogProb = F.log_softmax(x, dim = -1)
        # xExp = torch.exp(xLogProb)
        
        gsRes = F.gumbel_softmax(torch.log(x), tau = 0.1, hard = True, eps=1e-10, dim = -1)
        
        return gsRes # xLogProb, xExp
#########################################################################################################################
numFeatures = featMax.size()[1] # 992
numClasses = 2
model1 = MLP_GumbelSoftmax(numFeatures, numClasses)

if torch.cuda.device_count() > 1:
    model1 = nn.DataParallel(model1)
model1.cuda()
mask = model1(featMax) # mask.size() = (36,2)

ch = 32
for i in range(numBlock - 1):
    ch += 32 * (2 ** (i+1)) # 992
cp_ch = config.cp_channel # config.cp_channel = 800
grid_sizeB = 10 # config에 저장이 안됨
num_cpA = (config.grid_size**2) # 400
num_cpB = (grid_sizeB**2)       # 100
num_cp = num_cpA + num_cpB
num_cp_layer = config.num_cp_layer # config.num_cp_layer = 2

#########################################################################################################################
class MLP_CP_W(nn.Module):
    def __init__(self, ch, cp_ch, num_cp): # ch = 992, cp_ch = 800, num_cp = 500
        super().__init__()
        self.fc1 = nn.Linear(ch, cp_ch)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(cp_ch, num_cp * 4)

    def forward(self, x):
        # x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        # x = F.log_softmax(x, dim = -1)
        gsRes = F.gumbel_softmax(torch.log(x), tau = 0.1, hard = True, eps=1e-10, dim = -1)
        return gsRes
#########################################################################################################################

cp_w2 = MLP_CP_W(ch, cp_ch, num_cp)

if torch.cuda.device_count() > 1:
    cp_w2 = nn.DataParallel(cp_w2)
cp_w2.cuda()

output = cp_w2(featMax)

maskA = mask[..., 0].unsqueeze(1).expand(-1,num_cpA*4) # mask[..., 0] = (36,)    maskA = (36,1600)
maskB = mask[..., 1].unsqueeze(1).expand(-1,num_cpB*4) # mask[..., 1] = (36,)    maskB = (36,400)
maskAB = torch.cat((maskA, maskB), dim = 1) # maskAB = (36,2000)

outputMasked = output * maskAB
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

cpA = feats[..., : num_cpA*3] # 0~1200 
cpB = feats[..., num_cpA*4 : num_cpA*4 + num_cpB*3] # 1600 ~ 1900
cpAB = torch.cat((cpA, cpB), dim = 1 )
cp = torch.tanh(cpAB)

wA = feats[..., num_cpA*3 : num_cpA*4] # 1200 ~ 1600  
wB = feats[..., num_cpA*4 + num_cpB : ] # 1900 ~ 2000
wAB = torch.cat((wA, wB), dim = 1)
w = torch.sigmoid(wAB) + 1e-7
w = w.view(-1, config.grid_size + grid_sizeB, config.grid_size + grid_sizeB, 1)
                   
'''TRAINING'''
k = 0

best_e = 0
best_cd = 0
best_hd = 0
best_lap = 0

prev_test_cd = 1e8
check_time = time.time()

prev = 0
for e in range(1): # start_epoch, config.epochs + 1
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
        optimizer.zero_grad()
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
        print("\nx = points[:, l, :] : ", points[:, l, :].size())
        # print("output : ", output.size() )
        # print("weight : ", weight.size() )
        
        points_out, xyz_and_feats_out = Transformer.points, Transformer.xyz_and_feats
        
        points = points.permute(0, 2, 1)
        if anisotropic:
            scales, output, points, control_points = rescale_input_outputs(
                scales, output.view(config.batch_size, -1, 3), points, control_points,
                config.batch_size
            )
        
        ## loss를 구하기 위해 네트워크의 결과값인 output을 NURBS에 넣어 reconstructed points를 구한다
        output = output.view(-1, config.grid_size, config.grid_size, 3)
        output = torch.cat((output, weight), -1)
        output = nurbs(output)
        recon_prediction = output.view(-1, num_samples_u * num_samples_v, 3)
        
        ## loss를 구하는 과정
        cd = chamfer_distance(recon_prediction, points.permute(0, 2, 1))
        hd = haursdorff_distance(recon_prediction, points.permute(0, 2, 1))
        laplac_loss = laplacian_on_samples_loss(
            recon_prediction.reshape((config.batch_size, num_samples_u, num_samples_v, 3))
        )

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
        optimizer.step()

        pbar.set_description_str(desc="[Train] Epoch {}/{}, loss: {:.7f}".format(e, config.epochs, loss.item()), refresh=True)
        k += 1
       
        config_test = Con
        New_Backbone()
    ## validation 과정 준비
    test_cd = []
    test_hd = []
    test_lap = []
    Transformer.eval()
    
    ## validation 과정
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

        output = output.view(-1, config.grid_size, config.grid_size, 3)
        output = torch.cat((output, weight), -1)
        output = nurbs(output)
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
