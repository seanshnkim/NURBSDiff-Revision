import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.leaky_relu(bn(conv(new_points)))
#            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.leaky_relu(bn(conv(new_points)))
#            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.LeakyReLU(negative_slope=0.2),
#            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.LeakyReLU(negative_slope=0.2),
#            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2

################################################################################################################

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.LayerNorm(d_points),
            nn.Linear(d_points, d_model)
        )

        self.fc2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_points)
        )

        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
#            nn.GELU(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(d_model, d_model)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
#            nn.GELU(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
#        res = self.fc2(res) + pre
        return res, attn

################################################################################################################

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, d_points = cfg.num_point, cfg.num_block, cfg.num_neighbor, cfg.dim_input
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 32),
        )

        self.transformer1 = TransformerBlock(32, cfg.dim_transformer, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.dim_transformer, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3] # B X N X 3
        points = self.transformer1(xyz, self.fc1(x))[0] # B X N X 32

        xyz_and_feats = [(xyz, points)] # BN(C+3)
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))

        return points, xyz_and_feats # BNC, [BN3, BNC]

################################################################################################################  
class nurbs_world(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        self.npoints = cfg.num_point 
        self.nneighbor = cfg.num_neighbor
        #self.grid_sizeList = list(map(int, cfg.grid_size_list))       # [10,20,25,30]
        self.grid_sizeList = cfg.grid_size_list      # [10,20,25,30]
        self.num_net_classes = len(self.grid_sizeList) # 5 
        self.nblocks = cfg.num_block           # 5
        self.num_cp_layer = cfg.num_cp_layer   # 
        
        ch = 32
        for i in range(self.nblocks - 1):
            ch += 32 * (2 ** (i+1)) # 992
        self.ch = ch
        self.cp_ch = cfg.cp_channel # 800

        self.mlpSoftmax = nn.Sequential(nn.Linear(self.ch, 80, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(80, self.num_net_classes, bias=True),
                                        nn.Softmax(dim=-1)
                                       )
        
        self.mlpCP_W = []
        for nGridSize in cfg.grid_size_list:
            self.mlpCP_W.append(nn.Sequential(nn.Linear(self.ch, self.cp_ch),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Linear(self.cp_ch, (nGridSize**2)*4)  ) 
                               )
            
    def forward(self, x):      
        points, xyz_and_features = self.backbone(x)
        
        '''Extract featMax from features'''
        features = [ elem[1] for elem in xyz_and_features ]
        featMax = torch.max(features[0], dim=1).values
        for i in range(1,self.nblocks): # self.nblocks == 4
            temp = torch.max(features[i], dim=1).values
            featMax = torch.cat((featMax, temp), dim = 1) # featMax.size() = (36, 480)
        
        '''Probability of each network(featProbSum)'''
        featMaxProb = self.mlpSoftmax(featMax)
        
        '''Select a network(0~4) that has maximum probability from 36 batches'''
        _, indices = torch.max(featMaxProb, dim =1)
        
        featMaxList = []
        outputList = []
        cpList = []
        wList = []
        
        eachBatchSize = []
        
        for i in range(self.num_net_classes): # self.num_net_classes ==5
            eachBatchSize.append(featMax[indices == i].shape[0] )
            # If there is no network selected(no network that has maximum probability at all), skip the steps below
            if eachBatchSize[-1] == 0: 
                # If there is no network selected, just append empty list.
                #(This is definitely not an "elegant" ways of programming. Any better idea?)
                cpList.append(torch.tensor([]))
                wList.append(torch.tensor([]))
                continue
                
            featMaxList.append(featMax[indices == i])
            self.mlpCP_W[i].cuda()
            outputList.append(self.mlpCP_W[i](featMaxList[-1]))
            
            num_grid_size = self.grid_sizeList[i]
            
            cpList.append(  torch.tanh(outputList[-1][..., : (num_grid_size**2)*3] )  )
            
            w = torch.sigmoid(outputList[-1][...,   (num_grid_size**2)*3 : ] ) + 1e-4
            wList.append(w.view(-1, num_grid_size, num_grid_size, 1))

        self.eachBatchSize = eachBatchSize
        self.maxIndices = indices
        
        return cpList, wList


