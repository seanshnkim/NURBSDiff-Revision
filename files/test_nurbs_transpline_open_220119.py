import json
import logging
import os

import sys
import time
root="/home/cloudest/"
platform=root+"parsenet-codebase/"
nurbsDiff=root+"NURBSDiff/"
config_path = "./config_2022-01-19.yaml"

sys.path.append(platform)
sys.path.append(nurbsDiff)

from load_config import Config
from NURBSDiff.surf_eval import SurfEval as SurfEval
from NurbsNet_220119 import nurbs_world
from pytorch3d.loss import chamfer_distance

from shutil import copyfile
import shutil
import numpy as np
import open3d
import torch.optim as optim
import torch.utils.data
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.VisUtils import tessalate_points
from src.dataset import DataSetControlPointsPoisson
from src.dataset import generator_iter
from src.loss import (
    control_points_permute_closed_reg_loss,
    control_points_loss,
)
from src.loss import laplacian_loss, laplacian_on_samples_loss
from src.loss import (
    uniform_knot_bspline,
    spline_reconstruction_loss_one_sided,
    basis_function_one,
)
from src.utils import rescale_input_outputs, haursdorff_distance #, chamfer_distance

np.set_printoptions(precision=4)
config = Config(config_path)

userspace = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
logger.addHandler(handler)


# Model Loading
num_samples_u = 40
num_samples_v = 40
model_name = '2Nets_EachLoss_2022-01-19'
test_model_name = '2Nets_Test_20222-01-19'
Transformer = nurbs_world(config)

num_net_classes = len(config.grid_size_list)

nurbs_list = []

for i in range(num_net_classes):
    nurbs_list.append(SurfEval(config.grid_size_list[i], config.grid_size_list[i], dimension=3, p=3, q=3, 
                               knot_u=None, knot_v=None, out_dim_u=num_samples_u, out_dim_v=num_samples_v, 
                               method='tc', dvc='cuda'))


print("===================================================================================================================")
print("Training Model: {}\n".format(model_name))
print("Training Configure information: \"{}".format(config_path))
print("===================================================================================================================")

load_model = torch.load('saved_models/' + model_name + ".pth")
Transformer.load_state_dict(load_model['model_state_dict'])
Transformer.cuda()

for i in range(num_net_classes):
    nurbs_list[i].cuda()
    
split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

'''DATA LOADING'''
align_canonical = True
anisotropic = True
if_augment = False
if_save_meshes = False

dataset = DataSetControlPointsPoisson(
    platform + config.dataset_path,
    config.batch_size,
    splits=split_dict,
    size_v=20,
    size_u=20,
)

get_test_data = dataset.load_val_data(
    if_regular_points=True, align_canonical=align_canonical, anisotropic=anisotropic
)
loader = generator_iter(get_test_data, int(1e10))
get_test_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)

os.makedirs(
    "cad/gt/",
    exist_ok=True,
)
os.makedirs(
    "cad/prediction/",
    exist_ok=True,
)

'''TRAINING'''
best_e = 0
best_cd = 0
prev_test_cd = 1e8
fps_net = 0

test_cd = []
test_hd = []
test_lap = []

test_cd_list = [[] for n in range(num_net_classes)]
test_hd_list = [[] for n in range(num_net_classes)]
test_lap_list = [[] for n in range(num_net_classes)]

Transformer.eval()

for i in range(num_net_classes):
    nurbs_list[i].eval()

writer1 = SummaryWriter(root + "tensorboard/nurbs/transpline/" + test_model_name)
    

for test_b_id in range(config.num_test // config.batch_size):
    points_, parameters, control_points, scales, _ = next(get_test_data)[0]
    control_points = Variable(
        torch.from_numpy(control_points.astype(np.float32))
    ).cuda()

    points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    with torch.no_grad():
        l = np.arange(points.shape[1])
        np.random.shuffle(l)
        new_points = points[:, 0:config.num_point, :]

        check_time = time.time()
        
        output, weight, netInd = Transformer(points[:, l, :])
        
        fps_net += time.time() - check_time
        points = points.permute(0, 2, 1)
        output = output.view(config.batch_size, -1, 3)
        if anisotropic:
            scales, output, points, control_points = rescale_input_outputs(scales, output, points, control_points,
                                                                                   config.batch_size)
        
        
        output = torch.cat((output.view(-1, config.grid_size_list[netInd], config.grid_size_list[netInd], 3), weight), -1)
        check_time = time.time()
        output = nurbs_list[netInd](output)
        fps_net += time.time() - check_time
        recon_prediction = output.view(-1, num_samples_u * num_samples_v, 3)

        cd = chamfer_distance(recon_prediction, points.permute(0, 2, 1))
        hd = haursdorff_distance(recon_prediction, points.permute(0, 2, 1))
        lap = laplacian_on_samples_loss(recon_prediction.view(-1, num_samples_u, num_samples_v, 3))
        
        for i in range(num_net_classes):
            if i == netInd:
                test_cd_list[i].append(cd[0].data.cpu().numpy())
                test_hd_list[i].append(hd.data.cpu().numpy())
                test_lap_list[i].append(lap.data.cpu().numpy())
                
#     print(
#         '''\rIteration {}/{}, CD:{:.6f}, HD:{:.6f}, LAP:{:.6f}
#            \rMean SplineNet fps: {:.6f}
#         '''.format(
#             test_b_id, config.num_test // config.batch_size - 1,
#             test_cd[-1], test_hd[-1], test_lap[-1], fps_net/(test_b_id+1)
#         ),
#         end="",
#     )
    for i in range(num_net_classes):    
        if 
        writer1.add_scalar("Network " + str(i) + "Chamfer Distance Loss", test_cd_list[i][-1], test_b_id) 
        writer1.add_scalar("Network " + str(i) + "Haursdorff Distance Loss", test_hd_list[i][-1], test_b_id) 
        writer1.add_scalar("Network " + str(i) + "Laplacian Loss", test_lap_list[i][-1], test_b_id)
        writer1.add_scalar("Network " + str(i) + "Probability Sum", probability_sum_val[i], test_b_id)

    if if_save_meshes:
        print('A shape is saved... CD: {}'.format(test_cd[-1]))
        ones = torch.ones((config.batch_size, config.grid_size, config.grid_size, 1), requires_grad=False).cuda()
        recon_input_points = torch.cat((control_points, ones), -1)
        recon_input_points = nurbs(recon_input_points).view(-1, num_samples_u * num_samples_v, 3).data.cpu().numpy()
        recon_prediction = recon_prediction.data.cpu().numpy()

        # Save the predictions.
        for b in range(config.batch_size):
            pred_mesh = tessalate_points(recon_prediction[b], num_samples_u, num_samples_v)
            pred_mesh.paint_uniform_color([1, 0, 0])
            gt_mesh = tessalate_points(recon_input_points[b], num_samples_u, num_samples_v)
            gt_mesh.paint_uniform_color([1, 0, 0])

            open3d.io.write_triangle_mesh(
                "cad/gt/{}_gt.ply".format(
                    test_b_id
                ),
                gt_mesh,
            )
            open3d.io.write_triangle_mesh(
                "cad/prediction/{}_pred.ply".format(
                    test_b_id
                ),
                pred_mesh,
            )
    writer1.close()



# print(
#     "Test CD Loss: {}, Test HD Loss: {},  Test Lap: {}".format(
#         np.mean(test_cd), np.mean(test_hd), np.mean(test_lap)
#     )
# )
