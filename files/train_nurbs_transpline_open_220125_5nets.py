import json
import logging
import os

import sys
import time
from torch.utils.tensorboard import SummaryWriter

root="/home/cloudest/"
platform=root+"parsenet-codebase/"
nurbsDiff=root+"NURBSDiff/"
config_path = "./config_2022-01-19_5nets.yaml"

sys.path.append(platform)
sys.path.append(nurbsDiff)

from load_config import Config
from NURBSDiff.surf_eval import SurfEval
from NurbsNet_220125 import nurbs_world
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
from src.utils import rescale_input_outputs, haursdorff_distance  #, chamfer_distance

from torch.utils.tensorboard import SummaryWriter

################################################################################################################ 
'''MODEL NAME'''
model_name = '5Nets_2022-01-25_5Nets_Weighted'

################################################################################################################ 
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

################################################################################################################ 
plt.rcParams['figure.figsize'] = [13, 13]
plt.rcParams.update({'font.size': 25})

def colormap(data): # data: iteration X Flatten weight (88 X 400)
    plt.imshow(data, cmap='seismic', interpolation='nearest')
    plt.set_cmap('seismic')
    ax = plt.subplot()
    im = ax.imshow(data, cmap='seismic', interpolation='nearest')
    plt.yticks(np.arange(data.shape[0]))
    plt.xticks(np.arange(data.shape[1]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1%', pad=0.50)
    colorbar = plt.colorbar(im, cax=cax)
    colorbar_obj = plt.getp(colorbar.ax.axes, 'yticklabels')
    ax.invert_yaxis()
    ax.set_xlabel('U', fontsize='large', color='white', labelpad=20.0)
    ax.set_ylabel('V', fontsize='large', color='white', labelpad=20.0)
    ax.set_xticks(np.linspace(-0.51, config.grid_size-0.50, 21))
    ax.set_yticks(np.linspace(-0.51, config.grid_size-0.50, 21))
    ax.axes.xaxis.set_ticklabels([])
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(visible=True, color='gray')
    plt.setp(colorbar_obj, color='w')
    plt.show()
    
################################################################################################################ 
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

'''Transformer, nurbs, network class # defining'''
Transformer = nurbs_world(config)
num_net = len(config.grid_size_list)
nurbs_list = []

for i in range(num_net):
    nurbs_list.append(SurfEval(config.grid_size_list[i], config.grid_size_list[i], dimension=3, p=3, q=3,
                    out_dim_u=num_samples_u, out_dim_v=num_samples_v, method='tc', dvc='cuda'))

# 사용 가능한 GPU가 여러 개라면, 여러 개의 GPU를 활용, 데이터를 병렬 처리(DataParallel)하여 속도를 높인다.
if torch.cuda.device_count() > 1:
    Transformer = torch.nn.DataParallel(Transformer)
Transformer.cuda()
for i in range(num_net):
    nurbs_list[i].cuda()

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
        pin_memory=False,)
)

loader = generator_iter(get_val_data, int(1e10))
get_val_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,)
)

logger.info('Data Loading Complete')


'''OPTIMIZER SCHEDULER PREPARING'''
optimizer = optim.Adam(list(Transformer.parameters()) + list(nurbs_list[0].parameters()), lr=config.lr)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-5)
    

'''MODEL LOADING'''
try:
    checkpoint = torch.load('logs/trained_models/'+pretrain_model_path)
    start_epoch = checkpoint['epoch']
    Transformer.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Pretrain model has been successfully loaded')
except:
    logger.info('No existing model, starting training from scratch...')
    start_epoch = 0

    
'''PREPARING TENSORBOARD'''
writer1 = SummaryWriter(root + "tensorboard/nurbs/transpline/" + model_name)
        
'''TRAINING'''
k = 0

best_e = 0
best_cd = 0
best_hd = 0
best_lap = 0

prev_test_cd = 1e8
check_time = time.time()

prev = 0

for e in range(start_epoch, config.epochs + 1):
    probability_sum_train = torch.zeros(num_net, device = 'cuda:0')
    probability_sum_val = torch.zeros(num_net, device = 'cuda:0')
    train_hd = []
    train_cd = []
    train_lap = []
    Transformer.train()

    print("=" * 158)
    pbar = tqdm(range(config.num_train//config.batch_size), smoothing=0.9)
    for train_b_id in pbar:
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        points_, parameters, control_points, scales, _ = next(get_train_data)[0]
        control_points = Variable(
            torch.from_numpy(control_points.astype(np.float32))
        ).cuda()
        
        points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
        l = np.arange(config.num_point)
        np.random.shuffle(l)
        l = l[0:config.num_point]
        
        '''Transformer.forward()'''
        outputList, weightList = Transformer(points[:, l, :])
        
        # 각 네트워크별 probability 합을 계산. Backpropagation 과정에 포함되면 메모리 용량 초과 에러가 발생하므로
        # with torch.no_grad()를 쓴다.
#         with torch.no_grad():
#             probability_sum_train += Transformer.featProbSum
        points = points.permute(0, 2, 1)
        
        lossTotal = 0
        # cdTotal = torch.zeros(config.batch_size, 1) # cdTotal.shape = [36, 1]
        cdTotal = 0
        hdTotal = torch.zeros(config.batch_size, 1) # cdTotal.shape = [36, 1]
        laplac_lossTotal = torch.zeros(config.batch_size, 1) # cdTotal.shape = [36, 1]
        
        for i in range(num_net):
            if anisotropic: 
                scales = torch.tensor(scales, dtype = torch.float32).cuda()
                # print('\nscales.device :\n', scales.device)
                scales_rescaled, outputList[i], points_rescaled, control_points_rescaled = \
                rescale_input_outputs(scales, outputList[i].view(config.batch_size, -1, 3), points, control_points, config.batch_size)
                
            output = outputList[i].view(-1, Transformer.grid_sizeList[i], Transformer.grid_sizeList[i], 3)
            output = torch.cat((output, weightList[i]), -1)
            output = nurbs_list[i](output)
            recon_prediction = output.view(-1, num_samples_u * num_samples_v, 3) # num_samples_u, v = 40, 40
        
            '''LOSS CALCULATION'''
            # For chamfer distance(cd), custom method does not work due to memory overload.
            # For cd only it will not be multiplied by Transformer.featMaxProb.
            cd = chamfer_distance(recon_prediction, points_rescaled.permute(0, 2, 1))
            hd = haursdorff_distance(recon_prediction, points_rescaled.permute(0, 2, 1)) # hd.shape = [36]
            laplac_loss = laplacian_on_samples_loss( 
                recon_prediction.reshape((config.batch_size, num_samples_u, num_samples_v, 3)) ) # laplac_loss.shape = [36]
            
            cdTotal += cd[0]
            
            if i == 0:
                # cdTotal = cd.unsqueeze(1)
                hdTotal = hd.unsqueeze(1)
                laplac_lossTotal = laplac_loss.unsqueeze(1)
            else:
                # cdTotal = torch.cat((cd.unsqueeze(1), cdTotal), dim = 1)
                hdTotal = torch.cat((hd.unsqueeze(1), hdTotal), dim = 1)
                laplac_lossTotal = torch.cat((laplac_loss.unsqueeze(1), laplac_lossTotal), dim = 1)

        # cdTotal.shape, hdTotal.shape, laplac_lossTotal.shape = [36, 5]
        
        # cdTotalWgtd = Transformer.featMaxProb * cdTotal
        hdTotalWgtd = Transformer.featMaxProb * hdTotal
        laplac_lossTotalWgtd = Transformer.featMaxProb * laplac_lossTotal
        
        # cdMean = torch.mean(cdTotalWgtd)
        #cdMean = cdTotal # / num_net
        cdMean = torch.div(cdTotal, num_net)
        #print('\ncdTotal type :\n', type(cdTotal))
        hdMean = torch.mean(hdTotalWgtd)
        #print('\nhdTotal type :\n', type(hdMean))
        laplac_lossMean = torch.mean(laplac_lossTotalWgtd)
        #print('\nlaplac_lossMean type :\n', type(laplac_lossMean))
        
        lossTotal = cdMean + hdMean + 0.1 * laplac_lossMean
        lossTotal.backward(retain_graph=True)
        
        train_cd.append(cdMean.data.cpu().numpy())
        train_hd.append(hdMean.data.cpu().numpy())
        train_lap.append(laplac_lossMean.data.cpu().numpy())

        optimizer.step()
        
        pbar.set_description_str(desc="[Train] Epoch {}/{}, loss: {:.7f}".format(e, config.epochs, lossTotal.item()), refresh=True)

    '''VALIDATIONN'''
    test_cd = []
    test_hd = []
    test_lap = []
    Transformer.eval()
    
#     # 각 network 별로 loss를 저장하기 위한 변수
#     chamferDistList = []
#     hausdorffDistList = []
#     laplacianLossList = []
    
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
            outputList, weightList = Transformer(points[:, 0:config.num_point, :])
            # probability_sum_val += Transformer.featProbSum
            points = points.permute(0, 2, 1)
        
        lossTotal = 0
        # cdTotal = torch.zeros(config.batch_size, 1) 
        cdTotal = 0
        hdTotal = torch.zeros(config.batch_size, 1) 
        laplac_lossTotal = torch.zeros(config.batch_size, 1)
        
        for i in range(num_net):
            if anisotropic: 
                scales = torch.tensor(scales, dtype = torch.float32).cuda()
                
                scales_rescaled, outputList[i], points_rescaled, control_points_rescaled = \
                rescale_input_outputs(scales, outputList[i].view(config.batch_size, -1, 3), points, control_points, config.batch_size)
                
            output = outputList[i].view(-1, Transformer.grid_sizeList[i], Transformer.grid_sizeList[i], 3)
            output = torch.cat((output, weightList[i]), -1)
            output = nurbs_list[i](output)
            recon_prediction = output.view(-1, num_samples_u * num_samples_v, 3) # num_samples_u, v = 40, 40
        
            '''LOSS CALCULATION'''
            cd = chamfer_distance(recon_prediction, points_rescaled.permute(0, 2, 1))
            hd = haursdorff_distance(recon_prediction, points_rescaled.permute(0, 2, 1))
            laplac_loss = laplacian_on_samples_loss(
                recon_prediction.reshape((config.batch_size, num_samples_u, num_samples_v, 3)) )
            
            cdTotal += cd[0]
            
            if i == 0:
                # cdTotal = cd.unsqueeze(1)
                hdTotal = hd.unsqueeze(1)
                laplac_lossTotal = laplac_loss.unsqueeze(1)
            else:
                # cdTotal = torch.cat((cd.unsqueeze(1), cdTotal), dim = 1)
                hdTotal = torch.cat((hd.unsqueeze(1), hdTotal), dim = 1)
                laplac_lossTotal = torch.cat((laplac_loss.unsqueeze(1), laplac_lossTotal), dim = 1)
        
        # cdTotalWgtd = Transformer.featMaxProb * cdTotal
        hdTotalWgtd = Transformer.featMaxProb * hdTotal
        laplac_lossTotalWgtd = Transformer.featMaxProb * laplac_lossTotal

        cdMean = torch.div(cdTotal, num_net)
        hdMean = torch.mean(hdTotalWgtd)
        laplac_lossMean = torch.mean(laplac_lossTotalWgtd)
        
        lossTotal = cdMean + hdMean + 0.1 * laplac_lossMean
        lossTotal.backward(retain_graph=True) 
        
        test_cd.append(cdMean.data.cpu().numpy())
        test_hd.append(hdMean.data.cpu().numpy())
        test_lap.append(laplac_lossMean.data.cpu().numpy())
        
        pbar.set_description_str(desc="[Valid] loss: {:.7f}".format(lossTotal.item()), refresh=True)

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
#     with torch.no_grad():
#         colormap(Transformer.featMaxProb)
    
    writer1.add_scalar("Chamfer Distance Loss", test_cd[-1], e) # e -> iteration number
    writer1.add_scalar("Haursdorff Distance Loss", test_hd[-1], e) 
    writer1.add_scalar("Laplacian Loss", test_lap[-1], e)
    writer1.add_scalars("Batch Sizes of Each Network", {'Control Points 10': Transformer.eachBatchSize[0],
                                       'Control Points 20': Transformer.eachBatchSize[1],
                                       'Control Points 30': Transformer.eachBatchSize[2],
                                       'Control Points 40': Transformer.eachBatchSize[3],
                                       'Control Points 50': Transformer.eachBatchSize[4]}, e)
#     for i in range(num_net):
#         writer1.add_scalar("Probability Sum of Network " + str(i), probability_sum_val[i], e)

    

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

writer1.close()