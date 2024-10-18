import os
import torch
import numpy as np


def listdir_nohidden(path):
    mylist = [f for f in os.listdir(path) if not f.startswith('.')]
    return mylist


def normalize_trajectory(Rt, center='first', normalize_rotation=True):
    #assert center in ['first', 'mid'], 'center must be either "first" or "mid", got {}'.format(center)

    seq_len = Rt.shape[1]

    if center == 'first':
        origin_frame = 0
    elif center == 'mid':
        origin_frame = seq_len // 2
    else:
        # return unmodified Rt
        return Rt

    if normalize_rotation:
        origins = Rt[:, origin_frame : origin_frame + 1].expand_as(Rt).reshape(-1, 4, 4).inverse()
        normalized_Rt = torch.bmm(Rt.view(-1, 4, 4), origins)
        normalized_Rt = normalized_Rt.view(-1, seq_len, 4, 4)
    else:
        camera_pose = Rt.inverse()
        origins = camera_pose[:, origin_frame : origin_frame + 1, :3, 3]
        camera_pose[:, :, :3, 3] = camera_pose[:, :, :3, 3] - origins
        normalized_Rt = camera_pose.inverse()

    return normalized_Rt


def random_rotation_augment(trajectory_Rt):
    # given a trajectory, apply a random rotation
    angle = np.random.randint(-180, 180)
    angle = np.deg2rad(angle)
    _rand_rot = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    rand_rot = torch.eye(4)
    rand_rot[:3, :3] = torch.from_numpy(_rand_rot).float()

    for i in range(len(trajectory_Rt)):
        trajectory_Rt[i] = trajectory_Rt[i].mm(rand_rot)

    return trajectory_Rt

def list_to_tensor(data_list):
    if isinstance(data_list[0],np.ndarray):
        return torch.stack([torch.from_numpy(d) for d in data_list])
    else:
        return torch.stack(data_list)


def sim_to_data(Rts,Ks,rgbs,depths,normalize_traj=False,normalize_rot=False, calculate_mask=False, device='cuda'):
    Rts = list_to_tensor(Rts)
    Ks = list_to_tensor(Ks)
    rgbs = list_to_tensor(rgbs).permute(0,3,1,2)
    depths = list_to_tensor(depths).unsqueeze(1)

    if normalize_traj:
        Rts = Rts.unsqueeze(0)
        Rts = normalize_trajectory(Rts, center='first', normalize_rotation=normalize_rot)
        Rts = Rts[0]

    if calculate_mask:
        seq_len = len(Rts)
        padd_max_len = 500
        obs_mask = torch.zeros(seq_len)
        obs_mask[:len(rgbs)] = 1.0
        sorted_indices, seq_unique_list, seq_unique_counts, pose_map = calculate_mask_func(depths, Rts, Ks, seq_len=seq_len, padd_max_len=padd_max_len)
        padded_Rt = padd(Rts,seq_len)
        padded_Rt[torch.where(obs_mask==0)[0]] = torch.eye(4)
        padded_K = padd(Ks, seq_len)
        padded_K[torch.where(obs_mask==0)[0]] = Ks[0].float()
        return {'rgb': padd(rgbs, seq_len).to(device).unsqueeze(0), 'depth': padd(depths,seq_len).to(device).unsqueeze(0), 'Rt': padded_Rt.to(device).unsqueeze(0), 'K': padded_K.to(device).unsqueeze(0),
                'sorted_indices': padd(sorted_indices,seq_len).to(device).unsqueeze(0),
                'seq_unique_list': padd(padd(seq_unique_list,seq_len), padd_max_len).to(device).unsqueeze(0),
                'seq_unique_counts': padd(padd(seq_unique_counts,seq_len), padd_max_len).to(device).unsqueeze(0),
                'pose_map': pose_map.to(device).unsqueeze(0),
                'obs_mask': obs_mask.to(device).unsqueeze(0)}
    else:
        return {'rgb': rgbs.to(device).unsqueeze(0), 'depth': depths.to(device).unsqueeze(0), 'Rt': Rts.to(device).unsqueeze(0), 'K': Ks.to(device).unsqueeze(0)}


def padd(v, max_len, dim=0):
    if dim == 0:
        new_v = torch.full([max_len, *v.shape[1:]], 0.)
        new_v[:v.shape[0]] = v
    elif dim == 1:
        new_v = torch.full([v.shape[0], max_len, *v.shape[2:]], 0.)
        new_v[:, :v.shape[1]] = v
        #print(v.shape[1])
    return new_v

from models.gsn.common.nerf_utils import get_ray_bundle_batch
from torchvision.transforms import Resize
def calculate_mask_func(depth, Rt, K, seq_len=5, padd_max_len=2000, nerf_res=32, img_res=64, coordinate_scale=32, position_inspection_res=128):
    resize_render = Resize(nerf_res)
    w = h = z = position_inspection_res
    B = len(depth)

    downsampling_ratio = nerf_res / img_res
    fx, fy = K[0, 0, 0] * downsampling_ratio, K[0, 1, 1] * downsampling_ratio
    ray_origins, ray_directions = get_ray_bundle_batch(nerf_res, nerf_res, (fx, fy), Rt.inverse())

    depth_positions = resize_render(depth).view(B,-1, 1)
    positions_orig = ray_origins.view(B, -1, 3) + ray_directions.view(B, -1, 3) * depth_positions
    positions = torch.clamp(positions_orig/(coordinate_scale/2), -0.99, 0.99)

    obs_ids = list(range(len(depth)))

    scaled_xx = (positions[obs_ids, :, 0] * (w//2) + w//2).int()
    scaled_yy = (positions[obs_ids, :, 1] * (z//2) + z//2).int()
    scaled_zz = (positions[obs_ids, :, 2] * (h//2) + h//2).int()
    #print("orig ", positions[0, 0], scaled_xx[0,0], scaled_zz[0,0])

    pose_for_ordering = scaled_xx * (position_inspection_res) + scaled_zz
    sorted_pose, sorted_indices = torch.sort(pose_for_ordering)
    all_unique_list = torch.unique(sorted_pose)

    seq_unique_list, seq_unique_counts = [], []
    for i in range(len(sorted_pose)):
        unique_list, unique_counts = torch.unique(sorted_pose[i], return_counts=True)
        padded_unique_list = torch.cat((unique_list, torch.full(size=(len(all_unique_list) - len(unique_list),), fill_value=0.).to(unique_list.device)))
        padded_unique_counts = torch.cat((unique_counts, torch.full(size=(len(all_unique_list) - len(unique_list),), fill_value=0.).to(unique_list.device)))
        seq_unique_list.append(padded_unique_list)
        seq_unique_counts.append(padded_unique_counts)
    seq_unique_list = torch.stack(seq_unique_list)
    seq_unique_counts = torch.stack(seq_unique_counts)

    pose_map = torch.zeros([w, h])
    pose_map_flattend = pose_map.view(-1)
    pose_map_flattend[all_unique_list.long()] = 1.0

    mask = (positions_orig[:, :, 1] < -0.2) * (positions_orig[:, :, 1] > -0.3) * (depth_positions.squeeze(-1) < 9.9)
    pose_map_flattend[(pose_for_ordering * mask)[range(sorted_indices.shape[0]), sorted_indices].long()] = 2.0
    pose_map = pose_map_flattend.view(B, w, h)

    return sorted_indices, seq_unique_list, seq_unique_counts, pose_map


def calculate_mask_no_depth(Rt, K, seq_len=5, padd_max_len=2000, nerf_res=32, img_res=64, coordinate_scale=32, position_inspection_res=128):
    samples = 64
    resize_render = Resize(nerf_res)
    w = h = z = position_inspection_res
    B = len(Rt)
    downsampling_ratio = nerf_res /img_res
    fx, fy = K[0, 0, 0] * downsampling_ratio, K[0, 1, 1] * downsampling_ratio
    ray_origins, ray_directions = get_ray_bundle_batch(nerf_res, nerf_res, (fx, fy), Rt.inverse())
    positions = ray_origins.view(B, -1, 1, 3) + ray_directions.view(B, -1, 1, 3) * torch.linspace(0.,10.,64).view(1, 1, 64, 1).to(Rt.device)
    positions = torch.clamp(positions/(coordinate_scale/2), -0.99, 0.99)
    obs_ids = list(range(len(Rt)))

    scaled_xx = (positions[obs_ids, :, :, 0] * (w//2) + w//2).int()
    scaled_yy = (positions[obs_ids, :, :, 1] * (z//2) + z//2).int()
    scaled_zz = (positions[obs_ids, :, :, 2] * (h//2) + h//2).int()

    pose_for_ordering = scaled_xx * (position_inspection_res) + scaled_zz

    sorted_pose, sorted_indices = torch.sort(pose_for_ordering.view(B, -1))
    indices = torch.range(0, nerf_res*nerf_res-1).view(1,nerf_res*nerf_res,1).repeat(B,1,samples).view(B, -1).to(Rt.device)
    sorted_indices = torch.gather(indices, index=sorted_indices, dim=1)
    all_unique_list = torch.unique(sorted_pose)

    seq_unique_list, seq_unique_counts = [], []
    max_len = 0
    for i in range(len(sorted_pose)):
        unique_list, unique_counts = torch.unique(sorted_pose[i], return_counts=True)
        if max_len < len(unique_list):
            max_len = len(unique_list)
        padded_unique_list = torch.cat((unique_list, torch.full(size=(len(all_unique_list) - len(unique_list),), fill_value=0.).to(Rt.device)))
        padded_unique_counts = torch.cat((unique_counts, torch.full(size=(len(all_unique_list) - len(unique_list),), fill_value=0.).to(Rt.device)))
        seq_unique_list.append(padded_unique_list)
        seq_unique_counts.append(padded_unique_counts)
    seq_unique_list = torch.stack(seq_unique_list)[:, :max_len+1]
    seq_unique_counts = torch.stack(seq_unique_counts)[:, :max_len+1]

    return sorted_indices, seq_unique_list, seq_unique_counts, None