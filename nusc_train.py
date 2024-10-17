import time
import numpy as np
import torch
import argparse

import os.path as osp
import threading
from queue import Queue

from stl_d_lib import *
import utils
from utils import uniform, to_np, to_np_dict, dict_to_cuda, dist_between_two_cars_stack

import nusc_api as napi
from nusc_dataset import MyDataset
from nusc_model import Net
from nusc_viz import plot_nuscene_viz, plot_debug_scene, plot_paper_scene


def dup(x, m):  # (N, d) -> (N * m, d)
    return x.unsqueeze(1).repeat((1, m) + tuple(1 for xx in x.shape[1:])).reshape((-1,)+x.shape[1:])

def mask_mean(loss, mask, dim=None):
    if dim is not None:
        return torch.mean(loss * mask, dim=dim) / torch.clip(torch.mean(mask, dim=dim), 1e-2)
    else:
        return torch.mean(loss * mask) / torch.clip(torch.mean(mask), 1e-2)

def dynamics(s, u):
    x, y, th, v = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    w, a = u[..., 0], u[..., 1]
    dx = v * torch.cos(th)
    dy = v * torch.sin(th)
    dth = w
    dv = a
    ds = torch.stack([dx, dy, dth, dv], dim=-1)
    return ds

def generate_trajs(s, us, dt):  # (..., 4) x (..., T, 2) -> (..., T, 4)
    trajs = [s]
    assert s.shape[-1] == 4
    assert us.shape[-1] == 2
    assert us.shape[:-2] == s.shape[:-1]
    for ti in range(us.shape[-2]):
        curr_s = trajs[-1]
        ds = dynamics(curr_s, us[..., ti, :2])
        next_s = curr_s + ds * dt
        trajs.append(next_s)
    return torch.stack(trajs, dim=-2)

def get_neighbor_trajs(neighbors, nt, dt, full=False): # (N, k, 7) -> (N, k, T, 1+4)
    no_cmd = torch.zeros_like(neighbors[..., :2]).unsqueeze(-2).repeat(1, 1, nt-1, 1)
    neighbor_trajs = generate_trajs(neighbors[..., 1:5], no_cmd, dt)  # (N, K, T, 4)
    neighbor_valids = neighbors[..., 0:1].unsqueeze(-2).repeat(1, 1, nt, 1)
    if full:
        LWs = neighbors[..., 5:7].unsqueeze(-2).repeat(1, 1, nt, 1)
        neighbor_trajs_aug = torch.cat([neighbor_valids, neighbor_trajs, LWs], dim=-1)
    else:
        neighbor_trajs_aug = torch.cat([neighbor_valids, neighbor_trajs], dim=-1)
    return neighbor_trajs_aug

I_VAL = 0
I_X = 0
I_Y = 1
I_TH = 2
I_V = 3
I_VMIN = 0
I_VMAX = 1
I_DMIN = 2
I_DMAX = 3
I_DSAFE = 4
I_THMAX = 5

def prep_stl_cache(x, args):
    ego_L = args.ego_L
    ego_W = args.ego_W
    
    x["x2curr_d"], x["x2curr_th"] = napi.compute_t2l_dist(x["ego_traj"][..., I_X:I_Y+2], x["currlane_wpts"], args.clip_dist, with_angle=True, inline=args.inline)
    x["x2left_d"], x["x2left_th"] = napi.compute_t2l_dist(x["ego_traj"][..., I_X:I_Y+2], x["leftlane_wpts"], args.clip_dist, with_angle=True, inline=args.inline)
    x["x2right_d"], x["x2right_th"] = napi.compute_t2l_dist(x["ego_traj"][..., I_X:I_Y+2], x["rightlane_wpts"], args.clip_dist, with_angle=True, inline=args.inline)
    if args.collision_loss is not None:
        x["min_nei_d"], x["min_centroid_d"], x["radius_sum"] = compute_shortest_dist_refined(x["ego_traj"][..., I_X:I_X+6], x["neighbors"][..., I_X+1:I_X+6+1], x["neighbors"][..., I_VAL], 
                                                ego_L=ego_L, ego_W=ego_W, nL=args.refined_nL, nW=args.refined_nW, full=True)
    else:
        x["min_nei_d"] = compute_shortest_dist_refined(x["ego_traj"][..., I_X:I_X+6], x["neighbors"][..., I_X+1:I_X+6+1], x["neighbors"][..., I_VAL], 
                                                ego_L=ego_L, ego_W=ego_W, nL=args.refined_nL, nW=args.refined_nW)
    
    if args.norm_stl:
        x["v_factor"] = torch.clip((x["stlp"][..., I_VMAX] - x["stlp"][..., I_VMIN]), 0.3)  #/2
        x["d_factor"] = torch.clip((x["stlp"][..., I_DMAX] - x["stlp"][..., I_DMIN]) * 5, 0.3)  #/2
        x["safe_factor"] = torch.clip(x["stlp"][..., I_DSAFE], 0.3)
    
    return x
    
def build_stl_cache(args):
    nt = args.nt
    if args.norm_stl:
        keep_v_min = Always(0, nt, AP(lambda x: (x["ego_traj"][..., I_V] - x["stlp"][..., I_VMIN]) / x["v_factor"]))
        keep_v_max = Always(0, nt, AP(lambda x: (-x["ego_traj"][..., I_V] + x["stlp"][..., I_VMAX]) / x["v_factor"]))
        keep_d_min = Always(0, nt, AP(lambda x: (x["x2curr_d"] - x["stlp"][..., I_DMIN]) / x["d_factor"]))
        keep_d_max = Always(0, nt, AP(lambda x: (-x["x2curr_d"] + x["stlp"][..., I_DMAX]) / x["d_factor"] ))
        reach_right_d = Eventually(0, nt//2, Always(0, nt, 
            And(
                AP(lambda x: (x["x2right_d"] - x["stlp"][..., I_DMIN])/x["d_factor"]),
                AP(lambda x: (-x["x2right_d"] + x["stlp"][..., I_DMAX])/x["d_factor"]),
        )))
        reach_left_d = Eventually(0, nt//2, Always(0, nt, 
            And(
                AP(lambda x: (x["x2left_d"] - x["stlp"][..., I_DMIN])/x["d_factor"]),
                AP(lambda x: (-x["x2left_d"] + x["stlp"][..., I_DMAX])/x["d_factor"]),
        )))

        safe_list = [Always(0, nt, AP(lambda x: (x["min_nei_d"] - x["stlp"][..., I_DSAFE])/ x["safe_factor"]))]
    else:
        keep_v_min = Always(0, nt, AP(lambda x: x["ego_traj"][..., I_V] - x["stlp"][..., I_VMIN]))
        keep_v_max = Always(0, nt, AP(lambda x: -x["ego_traj"][..., I_V] + x["stlp"][..., I_VMAX]))
        keep_d_min = Always(0, nt, AP(lambda x: x["x2curr_d"] - x["stlp"][..., I_DMIN]))
        keep_d_max = Always(0, nt, AP(lambda x: -x["x2curr_d"] + x["stlp"][..., I_DMAX]))
        reach_right_d = Eventually(0, nt//2, Always(0, nt, 
            And(
                AP(lambda x: x["x2right_d"] - x["stlp"][..., I_DMIN]),
                AP(lambda x: -x["x2right_d"] + x["stlp"][..., I_DMAX]),
        )))
        reach_left_d = Eventually(0, nt//2, Always(0, nt, 
            And(
                AP(lambda x: x["x2left_d"] - x["stlp"][..., I_DMIN]),
                AP(lambda x: -x["x2left_d"] + x["stlp"][..., I_DMAX]),
        )))

        safe_list = [Always(0, nt, AP(lambda x: x["min_nei_d"] - x["stlp"][..., I_DSAFE]))]

    keep_th_max = Always(0, nt, AP(lambda x: (x["stlp"][..., I_THMAX] - x["x2curr_th"])/x["stlp"][..., I_THMAX]))
    reach_left_th = Eventually(0, nt//2, Always(0, nt, AP(lambda x: (x["stlp"][..., I_THMAX] -x["x2left_th"])/x["stlp"][..., I_THMAX])))
    reach_right_th = Eventually(0, nt//2, Always(0, nt, AP(lambda x: (x["stlp"][..., I_THMAX]-x["x2right_th"])/x["stlp"][..., I_THMAX])))

    stl_curr = ListAnd([keep_v_min, keep_v_max, keep_d_min, keep_d_max, keep_th_max] + safe_list)
    stl_left = ListAnd([keep_v_min, keep_v_max, reach_left_d, reach_left_th] + safe_list)
    stl_right = ListAnd([keep_v_min, keep_v_max, reach_right_d, reach_right_th] + safe_list)

    return [stl_curr, stl_left, stl_right]

def compute_shortest_dist_refined(state_a, state_b, ind, ego_L=None, ego_W=None, nL=4, nW=1, full=False):
    res = dist_between_two_cars_stack(state_a.unsqueeze(1), state_b, nL, nW, ego_L=ego_L, ego_W=ego_W, full=full)
    if full:
        car_dist, min_dist, rs1_rs2 = res
        return torch.min(torch.clip(car_dist, -5, 20) * ind + (1-ind) * 100, dim=1)[0], min_dist * ind + (1-ind) * 100, rs1_rs2
    else:
        return torch.min(torch.clip(res, -5, 20) * ind + (1-ind) * 100, dim=1)[0]

def get_stl_scores(scores_list, stl_i):
    return scores_list[0] * (stl_i==0).float() + scores_list[1] * (stl_i==1).float() + scores_list[2] * (stl_i==2).float() + scores_list[3] * (stl_i==3).float()

def get_dataloader(args):
    cache = nusc = nusc_map_d = None
    if args.offline:
        cache = np.load(utils.find_npz_path(args.cache_path), allow_pickle=True)["data"].item()
        meta_list = np.load(utils.find_npz_path(args.cache_path), allow_pickle=True)["meta_list"]
    else:
        nusc, nusc_map_d = napi.get_nuscenes(is_mini=args.mini)
        meta_list = napi.get_scene_tokens(nusc)
    data_len = len(meta_list)
    ridx = torch.arange(data_len)
    train_dataset = MyDataset(nusc, nusc_map_d, meta_list, cache, "train", args, ridx=ridx)
    if args.collect_data or args.check_stl_params:
        val_dataset = None
    else:
        val_dataset = MyDataset(nusc, nusc_map_d, meta_list, cache, "val", args, ridx=ridx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if val_dataset is None:
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return train_loader, val_loader, nusc, nusc_map_d, meta_list

def save_cache_data(batch, saved_sample_d):
    batch_np = to_np_dict(batch)
    for i in range(batch_np["traj_i"].shape[0]):
        traj_i = batch_np["traj_i"][i]
        ti = batch_np["ti"][i]
        if traj_i not in saved_sample_d:
            saved_sample_d[traj_i] = {}
        saved_sample_d[traj_i][ti] = {}
        for key in batch_np:
            if key != "params":
                saved_sample_d[traj_i][ti][key] = batch_np[key][i]
    return saved_sample_d

def collect_nuscene_data(train_loader, meta_list):
    saved_sample_d = {}
    for bi, batch in enumerate(train_loader):
        print("batch", bi, "total", len(train_loader))
        saved_sample_d = save_cache_data(batch, saved_sample_d)
    np.savez("%s/cache.npz"%(args.exp_dir_full), data=saved_sample_d, meta_list=meta_list)

def infer_gt_stlp(batch_cuda, gt_trajs, args, data_loader=None):
    DEFAULT_DMIN = -5
    DEFAULT_DMAX = 5
    DEFAULT_TH = 0.5
    gt_vmin = torch.min(gt_trajs[..., 3], dim=-1)[0]
    gt_vmax = torch.max(gt_trajs[..., 3], dim=-1)[0]
    neighbor_trajs_aug = batch_cuda["neighbor_trajs_aug"]
    nei_trajs = neighbor_trajs_aug[:, :, :, 1:7]
    nei_valid = neighbor_trajs_aug[:, :, :, 0]
    
    # (N, K, T)
    nei_dist = compute_shortest_dist_refined(gt_trajs[..., :6], nei_trajs[..., :6], nei_valid, ego_L=args.ego_L, ego_W=args.ego_W, nL=args.refined_nL, nW=args.refined_nW)
    gt_d_safe = torch.min(nei_dist, dim=-1)[0]  

    # dmin & dmax
    currlane_m = batch_cuda["currlane_wpts"]#.unsqueeze(1).repeat(1, args.nt, 1, 1)
    leftlane_m = batch_cuda["leftlane_wpts"]#.unsqueeze(1).repeat(1, args.nt, 1, 1)
    rightlane_m = batch_cuda["rightlane_wpts"]#.unsqueeze(1).repeat(1, args.nt, 1, 1)
    d_curr, th_curr = napi.compute_t2l_dist(gt_trajs[..., :3], currlane_m, args.clip_dist, inline=args.inline, with_angle=True)  # (N, T)
    d_left, th_left = napi.compute_t2l_dist(gt_trajs[..., :3], leftlane_m, args.clip_dist, inline=args.inline, with_angle=True)  # (N, T)
    d_right, th_right = napi.compute_t2l_dist(gt_trajs[..., :3], rightlane_m, args.clip_dist, inline=args.inline, with_angle=True)  # (N, T)

    highlevel = batch_cuda["gt_high_level"][:, 0]  # (N)
    gt_dmin0 = torch.min(d_curr, dim=-1)[0]
    gt_dmax0 = torch.max(d_curr, dim=-1)[0]
    gt_dmin1 = torch.min(d_left[:, args.nt//2-1:], dim=-1)[0]
    gt_dmax1 = torch.max(d_left[:, args.nt//2-1:], dim=-1)[0]
    gt_dmin2 = torch.min(d_right[:, args.nt//2-1:], dim=-1)[0]
    gt_dmax2 = torch.max(d_right[:, args.nt//2-1:], dim=-1)[0]

    gt_dmin = gt_dmin0 * (highlevel==0).float() + gt_dmin1 * (highlevel==1).float() + gt_dmin2 * (highlevel==2).float() + DEFAULT_DMIN * (highlevel==3).float()
    gt_dmax = gt_dmax0 * (highlevel==0).float() + gt_dmax1 * (highlevel==1).float() + gt_dmax2 * (highlevel==2).float() + DEFAULT_DMAX * (highlevel==3).float()

    gt_th0 = torch.max(th_curr, dim=-1)[0]
    gt_th1 = torch.max(th_left[:, args.nt//2-1:], dim=-1)[0]
    gt_th2 = torch.max(th_right[:, args.nt//2-1:], dim=-1)[0]
    gt_th_max = gt_th0 * (highlevel==0).float() + gt_th1 * (highlevel==1).float() + gt_th2 * (highlevel==2).float() + DEFAULT_TH * (highlevel==3).float()
    # print("gt_th_max", torch.min(gt_th_max), torch.mean(gt_th_max), torch.max(gt_th_max))
    if args.flex:
        return torch.stack([torch.clip(gt_vmin-1, -0.3), gt_vmax+1, gt_dmin-0.3, gt_dmax+0.3, torch.clip(gt_d_safe-0.1,0), gt_th_max+0.1], dim=-1)
    else:
        return torch.stack([gt_vmin-0.1, gt_vmax+0.1, gt_dmin-0.1, gt_dmax+0.1, gt_d_safe-0.1, gt_th_max+0.05], dim=-1)

def mul_n(x, n):  # (make n times in the 0-st dimension)
    xdim = x.dim()
    xshape = list(x.shape)
    return x[:, None].repeat(1, n, *[1]*(xdim - 1)).reshape(xshape[0]*n, *xshape[1:])

def pre_prepare_stl_cache(batch_cuda, dense_trajs=None, detach=False, repeat_n=None, mono=False, mono_n=None, gt_stlp=None):
    if mono:
        stl_input = {
            "neighbors": mul_n(batch_cuda["neighbors_traj"], mono_n),
            "currlane_wpts": mul_n(batch_cuda["currlane_wpts"], mono_n),  # (bs * m, n_segs, 3)
            "leftlane_wpts": mul_n(batch_cuda["leftlane_wpts"], mono_n),  # (bs * m, n_segs, 3)
            "rightlane_wpts": mul_n(batch_cuda["rightlane_wpts"], mono_n),  # (bs * m, n_segs, 3)
            "stlp": mul_n(gt_stlp, mono_n)[:, None, :],
            "dense_valids": mul_n(torch.ones_like(batch_cuda["gt_high_level"]), mono_n),
            "gt_high_level": mul_n(batch_cuda["gt_high_level"], mono_n),
        }
    else:
        stl_input = {
            "neighbors": batch_cuda["neighbors_dense"],
            "currlane_wpts": batch_cuda["currlane_wpts_dense"],  # (bs * m, n_segs, 3)
            "leftlane_wpts": batch_cuda["leftlane_wpts_dense"],  # (bs * m, n_segs, 3)
            "rightlane_wpts": batch_cuda["rightlane_wpts_dense"],  # (bs * m, n_segs, 3)
            "stlp": batch_cuda["stlp_dense"],
            "dense_valids": batch_cuda["valids_dense"],
            "gt_high_level": batch_cuda["gt_high_level"],
        }
    if detach:
        stl_input = {k:stl_input[k].detach() for k in stl_input}
    if repeat_n is not None:
        stl_input = {k:value.repeat(repeat_n, *[1] * (value.dim()-1)) for k,value in stl_input.items()}
    if dense_trajs is not None:
        stl_input["ego_traj"] = dense_trajs
    return stl_input

def compute_trajopt_loss_lite(dense_controls, dense_trajs, stls_cac, stl_input_cache, ii, opt_iters):
    bs, M, _, nt, _ = dense_trajs[:,:,:,:-1,:].shape
    stl_input_cache["ego_traj"] = dense_trajs[:,:,:,:-1,:].reshape(bs*M*3, nt, 4)
    dense_valids = stl_input_cache["dense_valids"]
    relu = torch.nn.ReLU()
    
    stl_input_cache = prep_stl_cache(stl_input_cache, args)
    res_list = [stl_i(stl_input_cache, args.smoothing_factor, full=True) for stl_i in stls_cac]
    scores_list = [res[0][:, 0].reshape(bs*M, 3)[:, i] for i,res in enumerate(res_list)]
    dense_scores = torch.stack(scores_list, dim=-1)  # (bs * M, 3)
    dense_loss = torch.mean(relu(args.stl_trajopt_thres-dense_scores) * dense_valids) / torch.clip(torch.mean(dense_valids), 1e-3)
    reg_loss = torch.mean(relu(dense_controls[..., 0]**2 - args.mul_w_max**2)) + torch.mean(relu(dense_controls[..., 1]**2 - args.mul_a_max**2))
    reg_loss = reg_loss * args.reg_loss
    trajopt_loss = dense_loss + reg_loss
    all_scores = evaluate_all_scores(dense_scores, stl_input_cache["gt_high_level"], stl_input_cache["dense_valids"])
    avg_speed = torch.mean(dense_trajs[..., :-1, 3])

    rd_tj = {}
    if args.measure_diversity:
        if ii==opt_iters-1:
            ma_std, ma_vol, ma_std_list, ma_vol_list = napi.measure_diversity(dense_trajs[..., :-1, :2].reshape(bs, args.n_randoms, 3, args.nt*2), 
                    dense_scores.reshape(bs, args.n_randoms, 3), dense_valids.reshape(bs, args.n_randoms, 3), args.nt)
            rd_tj["ma_std"] = torch.tensor([ma_std]).float().to(dense_trajs.device)
            rd_tj["ma_vol"] = torch.tensor([ma_vol]).float().to(dense_trajs.device)
            rd_tj["ma_std_list"] = torch.from_numpy(np.stack(ma_std_list, axis=-1)).float().to(dense_trajs.device)
            rd_tj["ma_vol_list"] = torch.from_numpy(np.stack(ma_vol_list, axis=-1)).float().to(dense_trajs.device)

    scene_acc = torch.mean((torch.max(dense_scores.reshape(bs, args.n_randoms, 3), dim=1)[0]>=0).float() * (dense_valids.reshape(bs, args.n_randoms, 3)[:,0]), dim=0) / torch.clip(torch.mean(dense_valids.reshape(bs, args.n_randoms, 3)[:,0], dim=0), 1e-3)

    return trajopt_loss, dense_loss, reg_loss, torch.mean((dense_scores>=0).float() * dense_valids) / torch.clip(torch.mean(dense_valids), 1e-3), avg_speed, dense_scores, dense_valids, all_scores, rd_tj, scene_acc

def compute_stl_dense(stl_input, stls_cac, stl_idx, mask, args, debug=False, tj_scores=None, scene=False):
    stl_input = prep_stl_cache(stl_input, args)
    res_list = [stl_i(stl_input, args.smoothing_factor, full=True) for stl_i in stls_cac]
    scores_list = [res[0][:, 0] for res in res_list]
    scores_list.append(scores_list[-1].detach() * 0.0 + 1.0)  # for outliers
    scores = get_stl_scores(scores_list, stl_idx[:, 0])
    mask_flat = mask.reshape(-1,)
    if args.oracle_filter and tj_scores is not None:
        tj_scores_cube = tj_scores.reshape(-1, args.n_randoms, 3)
        tj_scores_cube = torch.max(tj_scores_cube, dim=1, keepdim=True)[0]
        tj_val_rep = ((tj_scores_cube > 0).float()).repeat(1, args.n_randoms, 1)
        tj_val_flat = tj_val_rep.reshape(-1, )
        acc = mask_mean((scores>0).float(), mask_flat * tj_val_flat)
    else:
        acc = mask_mean((scores>0).float(), mask_flat)
    if debug:
        return scores_list, scores, acc, stl_input
    else:
        if scene:
            # 24576
            # print(mask.shape)
            scores_cube = scores.reshape(-1,args.n_randoms,3)
            mask_cube = mask.reshape(-1,args.n_randoms,3)
            # print(scores.shape, mask_flat.shape)
            scene_acc = mask_mean((torch.max(scores_cube,dim=1)[0]>0).float(), mask_cube[:,0,:])
            return scores_list, scores, acc, scene_acc
        else:
            return scores_list, scores, acc

def evaluate_all_scores(scores, gt_labels, valid_mask):
    bs = gt_labels.shape[0]
    all_scores = {x:[] for x in ["in_label_scores", "out_label_scores", 
                        "in_label_curr_scores", "in_label_left_scores", "in_label_right_scores", 
                        "out_label_curr_scores", "out_label_left_scores", "out_label_right_scores"]}
    in_inv_s = {0:"in_label_curr_scores", 1:"in_label_left_scores", 2:"in_label_right_scores"}
    out_inv_s = {0:"out_label_curr_scores", 1:"out_label_left_scores", 2:"out_label_right_scores"}
    scores_3d = scores.reshape(bs, args.n_randoms, 3)
    valid_mask_2d = valid_mask.reshape(bs * args.n_randoms, 3)
    valid_mask_3d = valid_mask_2d.reshape(bs, args.n_randoms, 3)
    for i in range(bs):
        if gt_labels[i]<3:
            for j in range(3):
                if valid_mask_3d[i, 0, j]>0:
                    if gt_labels[i]==j:
                        all_scores["in_label_scores"].append(scores_3d[i,:,j].detach())
                        all_scores[in_inv_s[j]].append(scores_3d[i,:,j].detach())
                    else:
                        all_scores["out_label_scores"].append(scores_3d[i,:,j].detach())
                        all_scores[out_inv_s[j]].append(scores_3d[i,:,j].detach())
    
    return all_scores

def compute_policy_loss(batch_cuda, nn_stlp, stls_cac, nn_trajs, rect_trajs, dense_trajs, args, 
                        diffusion_extras=None, vae_extras=None, dbgs_extras=None, bc_extras=None, nn_controls_adj=None, nn_controls_list_adj=None, opt_controls=None):
    bs = batch_cuda["ego_traj"].shape[0]
    neighbor_trajs_aug = batch_cuda["neighbor_trajs_aug"]
    relu = torch.nn.ReLU()
    
    rd = {}

    multi_gen = any([diffusion_extras is not None, vae_extras is not None, bc_extras is not None])

    self_trajs = nn_trajs if (args.rect_head==False) else rect_trajs

    stl_input = {"ego_traj": self_trajs[:, :-1]}
    if multi_gen:
        for key in ["neighbors", "currlane_wpts", "leftlane_wpts", "rightlane_wpts", "stlp"]:
            stl_input[key] = batch_cuda["%s_dense"%key]
        stl_input["dense_valids"] = batch_cuda["valids_dense"].reshape(-1, )
    else:
        stl_input["neighbors"]=neighbor_trajs_aug
        for key in ["currlane_wpts", "leftlane_wpts", "rightlane_wpts", "stlp"]:
            stl_input[key] = batch_cuda[key]
    
    stl_input_gt = {
        "ego_traj": batch_cuda["ego_traj"],
        "neighbors": neighbor_trajs_aug,
        "currlane_wpts": batch_cuda["currlane_wpts"],
        "leftlane_wpts": batch_cuda["leftlane_wpts"],
        "rightlane_wpts": batch_cuda["rightlane_wpts"],
        "stlp": batch_cuda["stlp"],  #.repeat(1, args.nt, 1),
    }
    rd["avg_speed"] = torch.mean(self_trajs[..., :-1, 3])
    rd["avg_speed_gt"] = torch.mean(batch_cuda["ego_traj"][..., 3])
    
    all_scores = None

    if multi_gen:
        valid_mask = stl_input["dense_valids"]
        scores_list, scores, acc = compute_stl_dense(stl_input, stls_cac, batch_cuda["highlevel_dense"], stl_input["dense_valids"], args)
        scores_list_gt, scores_gt, acc_gt = compute_stl_dense(stl_input_gt, stls_cac, batch_cuda["gt_high_level"], (batch_cuda["gt_high_level"][:, 0]!=3).float(), args)
        
        all_scores = evaluate_all_scores(scores, batch_cuda["gt_high_level"], valid_mask)
        rd["loss_stl"] = mask_mean(relu(args.stl_nn_thres - scores), valid_mask) * args.stl_weight
        rd["acc"] = acc
        rd["acc_gt"] = acc_gt
        rd["scores"] = rd["scores_all"] = scores
        rd["scores_gt"] = rd["scores_gt_all"] = scores_gt
        if args.collision_loss is not None:
            # [ref](https://github.com/zhejz/TrafficBots/blob/01a367db1ab7b353d50e98d9bfd1ac371d5f4848/src/utils/rewards.py#L112)
            coll_dist = torch.nn.ReLU()(1 - stl_input["min_centroid_d"] / torch.clip(stl_input["radius_sum"], 1e-1))
            coll_loss = torch.mean(torch.clip(torch.sum(coll_dist, dim=-1), max=1))
            rd["loss_coll"] = coll_loss * args.collision_loss
        else:
            rd["loss_coll"] = 0 * rd["loss_stl"]

        if args.measure_diversity:
            ma_std, ma_vol, ma_std_list, ma_vol_list = napi.measure_diversity(self_trajs[:, :-1, :2].reshape(bs, args.n_randoms, 3, args.nt*2), 
                scores.reshape(bs, args.n_randoms, 3), valid_mask.reshape(bs, args.n_randoms, 3), args.nt)
            rd["ma_std"] = torch.tensor([ma_std]).float().to(acc.device)
            rd["ma_vol"] = torch.tensor([ma_vol]).float().to(acc.device)
            rd["ma_std_list"] = torch.from_numpy(np.stack(ma_std_list, axis=-1)).float().to(acc.device)
            rd["ma_vol_list"] = torch.from_numpy(np.stack(ma_vol_list, axis=-1)).float().to(acc.device)

        if diffusion_extras is not None:
            noised_cmds_a, est_cmds_a, highlevel_dense, dense_scores, dense_valids, epi, raw_noise, nn_controls, steps, rect_controls = diffusion_extras
            gt_noise = raw_noise
            if args.stl_bc_mask:
                stl_acc_mask = (dense_scores * dense_valids > 0).float().reshape(bs * args.n_randoms * 3, 1, 1)
                rd["loss_diffusion"] = mask_mean(torch.square(gt_noise-est_cmds_a), stl_acc_mask.squeeze(-1)) 
            else:
                rd["loss_diffusion"] = torch.mean(torch.square(gt_noise-est_cmds_a))                     
            
            if args.rect_head:
                if args.diverse_loss:
                    # input: rect_controls, scores
                    NS = args.n_shards

                    samples = rect_controls.reshape(bs, args.n_randoms, 3, args.nt * 2).permute(0, 2, 1, 3).reshape(bs * 3 * NS, args.n_randoms // NS, args.nt, 2)
                    normal_x = torch.tensor([args.mul_w_max, args.mul_a_max]).float().to(samples.device)
                    samples = (samples / normal_x).reshape(bs * 3 * NS, args.n_randoms // NS, args.nt * 2)
                    quality = scores.reshape(bs, args.n_randoms, 3).permute(0, 2, 1).reshape(bs * 3 * NS, args.n_randoms // NS)
                    
                    dist = torch.norm(samples[:, :, None] - samples[:, None, :], dim=-1)  # (bs*4*3, N/4, N/4)
                    sim = torch.exp(-args.diversity_scale * dist)
                    if args.diverse_detach:
                        q_val = (quality>0).float().detach()
                    else:
                        q_val = (torch.exp(quality)) * ((quality>0).float())
                    q_mat = torch.diag_embed(q_val, dim1=-2, dim2=-1)

                    nL = torch.bmm(torch.bmm(q_mat, sim), q_mat)

                    I = torch.eye(args.n_randoms // NS)[None, :].to(nL.device)
                    nL_I_inv = torch.inverse(nL + I)
                    diversity = torch.einsum('bii->b', I - nL_I_inv)
                    loss_diversity = torch.mean(-diversity)
                    rd["loss_diversity"] = loss_diversity * args.diversity_weight
                    rd["loss_reg"] = mask_mean(torch.square(rect_controls-nn_controls.detach()), (scores[:,None,None]>=0).float()) 
                    rd["loss"] = rd["loss_stl"] + rd["loss_reg"] * args.rect_reg_loss + rd["loss_diversity"]
                else:                    
                    rd["loss_reg"] = torch.mean(torch.square((rect_controls[...,0]-nn_controls[...,0].detach())/args.mul_w_max)) + \
                                        torch.mean(torch.square((rect_controls[...,1]-nn_controls[...,1].detach())/args.mul_a_max))
                    rd["loss_reg"] = rd["loss_reg"] * args.rect_reg_loss
                    if args.extra_rect_reg is not None:
                        rd["extra_loss_reg"] = (torch.mean(relu((rect_controls[..., 0]/args.mul_w_max)**2 - 1)) + torch.mean(relu((rect_controls[..., 1]/args.mul_a_max)**2 - 1)))
                        rd["extra_loss_reg"] = rd["extra_loss_reg"] * args.extra_rect_reg
                    else:
                        rd["extra_loss_reg"] = rd["loss_reg"] * 0
                    
                    rd["loss"] = rd["loss_stl"]  + rd["loss_reg"] + rd["extra_loss_reg"] + rd["loss_coll"]
            else:
                rd["loss"] = rd["loss_stl"] + rd["loss_diffusion"] + rd["loss_coll"]

        elif vae_extras is not None:
            nn_controls, dense_controls, dense_trajs, highlevel_dense, dense_scores, dense_valids, (latent_mean, latent_logstd, latent_std), epi = vae_extras
            dense_controls_flat = dense_controls.reshape(-1, args.nt, 2)
            nn_controls_flat = nn_controls.reshape(-1, args.nt, 2)
            if args.stl_bc_mask:
                stl_acc_mask = (dense_scores * dense_valids > 0).float().reshape(bs * args.n_randoms * 3, 1, 1)
                rd["loss_vae_bc"] = mask_mean(torch.square(nn_controls_flat[:, :-1, :2] - dense_controls_flat[..., :-1, :2]), stl_acc_mask) * args.weight_vae_bc
            else:
                rd["loss_vae_bc"] = torch.mean(torch.square(nn_controls_flat[:, :-1, :2] - dense_controls_flat[..., :-1, :2])) * args.weight_vae_bc
            rd["loss_vae_bc"] = rd["loss_vae_bc"] * args.bc_weight
            rd["loss_vae_kl"] = (-1/2 * torch.mean(1 + 2*latent_logstd - latent_mean*latent_mean - latent_std*latent_std)) * args.weight_vae_kl
            rd["loss"] = rd["loss_stl"] + rd["loss_vae_bc"] + rd["loss_vae_kl"] + rd["loss_coll"]
            
        elif bc_extras is not None:
            nn_controls, dense_controls, dense_trajs, highlevel_dense, dense_scores, dense_valids, epi = bc_extras
            dense_controls_flat = dense_controls.reshape(-1, args.nt, 2)
            nn_controls_flat = nn_controls.reshape(-1, args.nt, 2)
            if args.stl_bc_mask:
                stl_acc_mask = (dense_scores * dense_valids > 0).float().reshape(bs * args.n_randoms * 3, 1, 1)
                rd["loss_bc"] = mask_mean(torch.square(nn_controls_flat[:, :-1, :2] - dense_controls_flat[..., :-1, :2]), stl_acc_mask)
            else:
                rd["loss_bc"] = torch.mean(torch.square(nn_controls_flat[:, :-1, :2] - dense_controls_flat[..., :-1, :2])) 
            rd["loss_bc"] = rd["loss_bc"] * args.bc_weight
            rd["loss"] = rd["loss_stl"] + rd["loss_bc"] + rd["loss_coll"]
        else:
            raise NotImplementedError
    else:
        valid_mask = (batch_cuda["gt_high_level"][:, 0]!=3).float()
        scores_list, scores, acc = compute_stl_dense(stl_input, stls_cac, batch_cuda["gt_high_level"], valid_mask, args)
        scores_list_gt, scores_gt, acc_gt = compute_stl_dense(stl_input_gt, stls_cac, batch_cuda["gt_high_level"], valid_mask, args)
        ##### masked acc filtered out outliers        
        rd["loss_bc"] = torch.mean(torch.square(nn_trajs[:, :-1, :2] - batch_cuda["ego_traj"][..., :2]))
        rd["loss_bc_masked"] = mask_mean(torch.square(nn_trajs[:, :-1, :2] - batch_cuda["ego_traj"][..., :2]), (1-valid_mask)[..., None, None])
        rd["loss_stl"] = mask_mean(relu(args.stl_nn_thres - scores), valid_mask)

        if args.bc:
            rd["loss"] = rd["loss_bc"]
        else:
            rd["loss"] = rd["loss_stl"]    
        rd["acc"] = acc
        rd["acc_gt"] = acc_gt
        rd["scores"] = scores
        rd["scores_gt"] = scores_gt

    return rd, all_scores

def get_diffusion_coeffs(args):
    if args.cos:
        t = torch.linspace(0, 1, args.diffusion_steps+1)
        alpha_bar = torch.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        beta = torch.clip(1-alpha_bar[1:]/alpha_bar[:-1], 0, 0.999) * 0.2
    else:
        beta = torch.linspace(args.beta_start, args.beta_end, args.diffusion_steps)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return (beta.cuda(), alpha.cuda(), alpha_hat.cuda())

def diffusion_prep(dense_controls, n_randoms, coeffs=None, mono=False):
    if mono:
        n = dense_controls.shape[0] * n_randoms
        mimic_cmd = dense_controls[:,None].repeat(1, n_randoms, 1, 1).reshape(n, args.nt, 2)
    else:
        n = dense_controls.shape[0] * n_randoms * 3
        mimic_cmd = dense_controls.reshape(n, args.nt, 2)
    mimic_cmd_w = mimic_cmd[..., 0] / args.mul_w_max
    mimic_cmd_a = mimic_cmd[..., 1] / args.mul_a_max
    mimic_cmd = torch.stack([mimic_cmd_w, mimic_cmd_a], dim=-1)
    mimic_cmd = mimic_cmd.reshape(n, args.nt * 2)
    noise = torch.normal(0, 1, (n, args.nt * 2)).cuda().float()
    beta, alpha, alpha_hat = coeffs
    t = torch.randint(low=1, high=args.diffusion_steps, size=(n, )).to(mimic_cmd.device)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
    return noise, t[:, None], None, sqrt_alpha_hat * mimic_cmd + sqrt_one_minus_alpha_hat * noise

def diffusion_rollout(noise, net, batch_cuda, highlevel_dense, feature, args, 
                      coeffs=None, fastforward=False, n_randoms=None, return_feature=False, mono=False, tmp_stlp=None, guidance_extras=None, maximize=False):
    n = noise.shape[0]
    self_beta, self_alpha, self_alpha_hat = coeffs
    net.eval()
    with torch.set_grad_enabled(args.grad_rollout and args.rect_head==False):
        x = torch.randn_like(noise)
        t = (torch.ones(n) * args.diffusion_steps-1).long().to(noise.device)
        res_list = [x]

        if fastforward==False:
            for i in reversed(range(1, args.diffusion_steps)):
                x = res_list[-1]
                if mono:
                    ext = {"timestep":t[:, None], "highlevel": highlevel_dense, "noise": x, "stlp": tmp_stlp, "gt_stlp":tmp_stlp}
                    predicted_noise = net(batch_cuda, ext=ext, prev_feature=feature, n_randoms=n_randoms)
                else:
                    ext = {"timestep":t[:, None], "highlevel": highlevel_dense, "noise": x, "stlp": batch_cuda["stlp_dense"]}
                    if feature is None:
                        predicted_noise, feature = net(batch_cuda, ext=ext, get_feature=True, n_randoms=n_randoms)
                    else:
                        predicted_noise = net(batch_cuda, ext=ext, prev_feature=feature, n_randoms=n_randoms)
                predicted_noise = predicted_noise.reshape(n, args.nt*2)
                alpha = self_alpha[t][:, None]
                alpha_hat = self_alpha_hat[t][:, None,]
                beta = self_beta[t][:, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                mu = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                triggered_guidance=False
                if args.guidance:
                    i_val = args.diffusion_steps-1-i if args.guidance_reverse else i
                    if args.guidance_sets is not None:
                        if i_val in args.guidance_sets:
                            triggered_guidance = True
                    elif args.guidance_freq is not None:
                        if i_val % args.guidance_freq == 0:
                            triggered_guidance = True
                    elif i<=args.guidance_before:
                        triggered_guidance = True
                if triggered_guidance:
                    with torch.set_grad_enabled(True):
                        # assert args.test
                        new_batch, states_flat_new, stls_cac = guidance_extras
                        N = states_flat_new.shape[0]
                        relu = torch.nn.ReLU()
                        mu_init = mu.reshape(N, args.nt, 2)
                        mu_opt = mu_init.detach().requires_grad_()
                        optimizer = torch.optim.Adam([mu_opt], lr=args.guidance_lr)
                        for j in range(args.guidance_niters):
                            opt_w = mu_opt[..., 0] * args.mul_w_max
                            opt_a = mu_opt[..., 1] * args.mul_a_max
                            opt_u = torch.stack([opt_w, opt_a], dim=-1)
                            nn_trajs = generate_trajs(states_flat_new, opt_u, args.dt).reshape(N, args.nt+1, 4)
                            opt_input = pre_prepare_stl_cache(new_batch, dense_trajs=nn_trajs[:, :-1])
                            valid_mask = opt_input["dense_valids"]
                            scores_list, scores, acc = compute_stl_dense(opt_input, stls_cac, new_batch["highlevel_dense"], opt_input["dense_valids"], args)
                            if maximize:
                                loss = mask_mean(relu(100 - scores), valid_mask.reshape(-1))
                            else:
                                loss = mask_mean(relu(args.stl_nn_thres - scores), valid_mask.reshape(-1))

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            with torch.no_grad():
                                delta_u_clipped = torch.clip(torch.abs(mu_opt - mu_init), -beta[0,0].item(), beta[0,0].item())
                                mu_opt.data = mu_init + delta_u_clipped
                        mu = mu_opt.reshape(N, -1).detach()
                x = mu + torch.sqrt(beta) * noise
                t = t - 1
                res_list.append(x)
        
        diffused_result = normalize_diff(res_list[-1], n, args.nt, args.mul_w_max, args.mul_a_max, args.diffusion_clip)
        if args.diff_full:
            final_list = [normalize_diff(res, n, args.nt, args.mul_w_max, args.mul_a_max, args.diffusion_clip) for res in res_list[:-1]] + [diffused_result]

    if args.diff_full:
        if return_feature:
            return diffused_result, feature, final_list
        else:
            return diffused_result, final_list
    else:
        if return_feature:
            return diffused_result, feature
        else:
            return diffused_result

def normalize_diff(x, n, nt, w_max, a_max, clip):
    x = x.reshape(n, nt, 2)
    final_res_w = x[..., 0] * w_max
    final_res_a = x[..., 1] * a_max
    if clip:
        final_res_w = torch.clip(final_res_w, -w_max, w_max)
        final_res_a = torch.clip(final_res_a, -a_max, a_max)
    final_res = torch.stack([final_res_w, final_res_a], dim=-1)
    return final_res

def get_dense_stlp(batch_cuda, the_stlp, args, n_randoms=None):
    bs = the_stlp.shape[0]

    if n_randoms is None:
        n_randoms = args.n_randoms

    high_level = batch_cuda["gt_high_level"].reshape(bs, 1, 1)

    stlp_mul_mid = the_stlp.unsqueeze(1).repeat(1, n_randoms, 1)  # (bs, n_randoms, 6)

    # prior
    vmin = 0
    vmax = 20
    dmin = -2.5
    dmax = 2.5
    dsafe = 0.1
    thmax = 0.5

    def generate_flex_pstl(stlp_mul_mid, the_high_level=2):
        assert args.test or (args.trajopt_only and args.epochs==1)
        vd0 = uniform(1.3, 3, (bs, 1)).repeat(1, n_randoms).to(high_level.device)
        vd1 = uniform(1.3, 3, (bs, 1)).repeat(1, n_randoms).to(high_level.device)
        new_vmin = torch.clip(stlp_mul_mid[:, :, 0] - vd0, -0.3)
        new_vmax = torch.clip(stlp_mul_mid[:, :, 1] + vd1, -0.3)
        if the_high_level==0:
            lamb0 = uniform(0, 1, (bs, 1)).repeat(1, n_randoms).to(high_level.device)
            lamb1 = uniform(0, 1, (bs, 1)).repeat(1, n_randoms).to(high_level.device)
            new_dmin = lamb0 * stlp_mul_mid[:, :, 2] + (1 - lamb0) * (stlp_mul_mid[:, :, 2] - 2.5)
            new_dmax = lamb1 * stlp_mul_mid[:, :, 2] + (1 - lamb1) * (stlp_mul_mid[:, :, 2] + 2.5)
        else:
            new_dmin = uniform(-2.5, -0.5, (bs, 1)).repeat(1, n_randoms).to(high_level.device)
            new_dmax = uniform(0.5, 2.5, (bs, 1)).repeat(1, n_randoms).to(high_level.device)
        lamb2 = uniform(0, 1, (bs, 1)).repeat(1, n_randoms).to(high_level.device)
        new_dsafe = torch.clip(lamb2 * stlp_mul_mid[:, :, 4] + (1-lamb2) * (stlp_mul_mid[:, :, 4] - 1.5), 0)

        lamb3 = uniform(0, 1, (bs, 1)).repeat(1, n_randoms).to(high_level.device)
        new_thmax = lamb3 * stlp_mul_mid[:, :, 5] + (1-lamb3) * (stlp_mul_mid[:, :, 5] + 0.3)

        return torch.stack([new_vmin, new_vmax, new_dmin, new_dmax, new_dsafe, new_thmax], dim=-1)

    # fixed value case
    if args.flex:
        default_stlp0 = generate_flex_pstl(stlp_mul_mid, the_high_level=0)
        default_stlp1 = generate_flex_pstl(stlp_mul_mid, the_high_level=1)
        default_stlp2 = generate_flex_pstl(stlp_mul_mid, the_high_level=2)
        stlp_mul = torch.stack(
            [
                (high_level*(3-high_level)==0).float() * stlp_mul_mid + (high_level*(3-high_level)!=0).float() * default_stlp0,
                (high_level==1).float() * stlp_mul_mid + (high_level!=1).float() * default_stlp1,
                (high_level==2).float() * stlp_mul_mid + (high_level!=2).float() * default_stlp2,
            ], dim=-2)

    else:
        default_stlp = torch.tensor([vmin, vmax, dmin, dmax, dsafe, thmax]).to(the_stlp.device).reshape(1, 1, 6).repeat(bs, n_randoms, 1)
        # TODO sampled value case
        # off-label cases, should be initialized with 1. fixed values 2. random values within some ranges
        stlp_mul = torch.stack(
            [
                (high_level==0).float() * stlp_mul_mid + (high_level!=0).float() * default_stlp,
                (high_level==1).float() * stlp_mul_mid + (high_level!=1).float() * default_stlp,
                (high_level==2).float() * stlp_mul_mid + (high_level!=2).float() * default_stlp,
            ], dim=-2)

    stlp_mul = stlp_mul.reshape(bs * n_randoms * 3, 1, 6)

    return stlp_mul

def augment_batch_data(batch, the_stlp, args, n_randoms=None, stlp_dense=None):
    if n_randoms is None:
        new_sample=False
        n_randoms = args.n_randoms
    else:
        new_sample=True
        
    m = n_randoms * 3
    bs = batch["currlane_wpts"].shape[0]
    batch["neighbors_dense"] = dup(batch["neighbor_trajs_aug"], m)
    batch["currlane_wpts_dense"] = dup(batch["currlane_wpts"], m)
    batch["leftlane_wpts_dense"] = dup(batch["leftlane_wpts"], m)
    batch["rightlane_wpts_dense"] = dup(batch["rightlane_wpts"], m)
    batch["stlp"] = the_stlp.unsqueeze(-2)  # (bs, 1, 6)

    # (bs * m, 1, 6)
    if stlp_dense is not None:
        batch["stlp_dense"] = stlp_dense
    else:
        if args.load_stlp:
            if new_sample:
                batch["stlp_dense"] = batch["pre_stlp"].reshape(bs, args.n_randoms, 3, 6)[:, 0:1, :, :].repeat(1, args.sampling_size, 1, 1).reshape(bs*m, 1, 6)
            else:
                batch["stlp_dense"] = batch["pre_stlp"].reshape(bs * m, 1, 6)
        else:
            batch["stlp_dense"] = get_dense_stlp(batch, the_stlp, args, n_randoms=n_randoms)
    
    valids = torch.cat([batch["curr_id"],batch["left_id"],batch["right_id"]], dim=-1)  # (bs, 3)
    batch["valids_dense"] = dup(valids, n_randoms).reshape(bs * n_randoms, 3)
    batch["highlevel_dense"] = torch.tensor([0, 1.0, 2.0]).reshape(1, 3, 1).repeat(bs * n_randoms, 1, 1).reshape(bs * m, 1).cuda().float()
    return batch

def compute_avg_acc(all_scores):
    if len(all_scores)>0:
        return torch.mean((all_scores>0).float())
    else:
        return -1 

def print_all_scores(mode, all_scores_list):
    print("%5s   Inlabel Acc.:%.3f (%.3f %.3f %.3f)   Outlabel Acc.:%.3f (%.3f %.3f %.3f)"%(
        mode, 
        compute_avg_acc(all_scores_list["in_label_scores"]),
        compute_avg_acc(all_scores_list["in_label_curr_scores"]),
        compute_avg_acc(all_scores_list["in_label_left_scores"]),
        compute_avg_acc(all_scores_list["in_label_right_scores"]),
        compute_avg_acc(all_scores_list["out_label_scores"]),
        compute_avg_acc(all_scores_list["out_label_curr_scores"]),
        compute_avg_acc(all_scores_list["out_label_left_scores"]),
        compute_avg_acc(all_scores_list["out_label_right_scores"]),
    ))

def save_trajopt_params(params, iter_i, traj_i, ti, args, save_stlp=None):
    bs = params.shape[0]
    params_np = to_np(params)
    if save_stlp is not None:
        save_stlp_np = to_np(save_stlp).reshape(bs, args.n_randoms, 3, 1, save_stlp.shape[-1])
    # iter_i in ["init", "final", (0,1,2...)]
    if args.test==False:
        for i in range(bs):
            if iter_i=="scores":
                fpath = "scores_%05d_%04d.npy"%(traj_i[i], ti[i])
            elif iter_i=="init": # save the final one
                fpath = "params_%05d_%04d_init.npy"%(traj_i[i], ti[i])
            elif iter_i=="final":
                fpath = "params_%05d_%04d.npy"%(traj_i[i], ti[i])
            else:
                fpath = "params_%05d_%04d_iter%05d.npy"%(traj_i[i], ti[i], iter_i)
                
            full_path = osp.join(args.model_dir, fpath)
            np.save(full_path, params_np[i])
            if save_stlp is not None:
                fpath = "params_%05d_%04d_stlp.npy"%(traj_i[i], ti[i])
                full_path = osp.join(args.model_dir, fpath)
                np.save(full_path, save_stlp_np[i])

def add_to(viz_cache, mode, key, val):
    if mode not in viz_cache:
        viz_cache[mode] = {}
    if key not in viz_cache[mode]:
        viz_cache[mode][key] = []
    viz_cache[mode][key].append(val.detach().cpu())

def print_for(met_d, print_key_dict, just_avg=False):
    s = []
    for k1, k2 in print_key_dict.items():
        if k2 in met_d:
            if just_avg:
                s.append("%s: (%.3f)"%(k1, met_d(k2)))
            else:
                s.append("%s: %.3f(%.3f)"%(k1, met_d[k2], met_d(k2)))
    return " ".join(s)

def check_stl_params(data_loader, meta_list, stls_cac, args):
    traj_i_list=[]
    ti_list=[]
    highlevel_list=[]
    gt_stlp_list=[]
    scores_list=[]
    acc_list=[]

    nusc, nusc_map_d = napi.get_nuscenes(is_mini=args.mini)

    for bi, batch in enumerate(data_loader):
        batch_cuda = dict_to_cuda(batch)
        traj_i = batch_cuda["traj_i"]
        ti = batch_cuda["ti"]
        gt_trajs = batch_cuda["ego_traj"][..., :4]
        states = gt_trajs[..., 0, :4]

        bs = states.shape[0]
        batch_cuda["neighbor_trajs_aug"] = batch_cuda["neighbors_traj"][..., :7]
        gt_stlp = infer_gt_stlp(batch_cuda, gt_trajs, args, data_loader)
        batch_cuda["stlp"] = gt_stlp[:, None].repeat(1, args.nt, 1)
        gt_high_level = batch_cuda["gt_high_level"]

        stl_input_gt = {
            "ego_traj": batch_cuda["ego_traj"],
            "gt_traj": batch_cuda["ego_traj"],
            "neighbors": batch_cuda["neighbor_trajs_aug"],
            "currlane_wpts": batch_cuda["currlane_wpts"],
            "leftlane_wpts": batch_cuda["leftlane_wpts"],
            "rightlane_wpts": batch_cuda["rightlane_wpts"],
            "stlp": batch_cuda["stlp"],  #.repeat(1, args.nt, 1),
        }
        
        _, scores_gt, acc_gt, _ = compute_stl_dense(stl_input_gt, stls_cac, batch_cuda["gt_high_level"], (batch_cuda["gt_high_level"][:, 0]!=3).float(), args, debug=True)

        vmin = torch.min(gt_trajs[..., 3])
        vmax = torch.max(gt_trajs[..., 3])
        dmin = torch.min(gt_stlp[:, 2])
        dmax = torch.max(gt_stlp[:, 3])
        dsafe1 = torch.min(gt_stlp[:, 4])
        dsafe2 = torch.max(gt_stlp[:, 4])
        
        traj_i_list.append(traj_i)
        ti_list.append(ti)
        highlevel_list.append(gt_high_level)
        gt_stlp_list.append(gt_stlp)
        scores_list.append(scores_gt)
        acc_list.append(acc_gt)
        print("%03d/%03d/ ACC:%.3f vmin:%.3f vmax:%.3f dmin:%.3f dmax:%.3f dsafe:%.3f %.3f"%(
            bi, len(data_loader), torch.mean(acc_gt), 
            vmin, vmax, dmin, dmax, dsafe1, dsafe2,
            ))
    
    traj_i_list = torch.cat(traj_i_list, dim=0)
    ti_list = torch.cat(ti_list, dim=0)
    highlevel_list = torch.cat(highlevel_list, dim=0)
    gt_stlp_list = torch.cat(gt_stlp_list, dim=0)
    scores_list = torch.cat(scores_list, dim=0)
    acc_list = torch.stack(acc_list, dim=0)
    print("ACC:%.3f"%(torch.mean(acc_list)))

def compute_ade_fde(gt_trajs, est_trajs, mask):
    # gt_trajs (N, nt, 6)
    # est_trajs (N, m, 3, nt, 6)
    bs, nt, k = gt_trajs.shape
    mask = mask.reshape(bs, -1)
    est_trajs_cmp = est_trajs.reshape(bs, -1, nt, k)
    err_t = torch.sum(torch.square((gt_trajs[:, None]-est_trajs_cmp)*mask[:,:,None,None] + (1-mask[:,:,None,None])*10000), dim=-1)
    err = torch.mean(err_t, dim=-1)
    ade = torch.mean(torch.min(err, dim=-1)[0])
    fde = torch.mean(torch.min(err_t[:,:,-1], dim=-1)[0])
    return ade, fde


def run_sampling_test(stls_cac, data_loader, net, coeffs, args, result_queue, thread_nusc):
    vae_eta = None
    md = utils.MeterDict()
    nusc, nusc_map_d = None, None
    if args.time_profile:
        myt = utils.MyTimer()

    interest_list = [
        (5, 1),
        (6, 1),
        (19, 5),
        (76, 15),
        (96, 8),
        (303, 16)
    ]

    for bi, batch in enumerate(data_loader):
        if vae_eta is not None:
            vae_eta.update()
        if bi>args.n_trials:
            continue
        batch_cuda = dict_to_cuda(batch)
        gt_trajs = batch_cuda["ego_traj"][..., :4]
        states = gt_trajs[..., 0, :4]
        bs = states.shape[0]
        batch_cuda["neighbor_trajs_aug"] = batch_cuda["neighbors_traj"][..., :7]
        gt_stlp = infer_gt_stlp(batch_cuda, gt_trajs, args)
        batch_cuda = augment_batch_data(batch_cuda, gt_stlp, args)
        n = bs * args.n_randoms * 3

        # trajopt trajectories
        dense_states = states.unsqueeze(1).unsqueeze(1).repeat(1, args.n_randoms, 3, 1)       

        N = bs * args.sampling_size * 3
        feature = None

        dense_controls = batch_cuda["params"]
        dense_trajs = generate_trajs(dense_states, dense_controls, args.dt).reshape(n, args.nt+1, 4)
        tj_stl_input = pre_prepare_stl_cache(batch_cuda, dense_trajs=dense_trajs[:, :-1])

        _, tj_scores, tj_acc, tj_scene_acc = compute_stl_dense(tj_stl_input, stls_cac, batch_cuda["highlevel_dense"], tj_stl_input["dense_valids"], args, scene=True)

        tj_ma_std, tj_ma_vol, tj_ma_std_list, tj_ma_vol_list = napi.measure_diversity(
                dense_trajs[:, :-1, :2].reshape(bs, args.n_randoms, 3, args.nt*2), 
                tj_scores.reshape(bs, args.n_randoms, 3), 
                tj_stl_input["dense_valids"].reshape(bs, args.n_randoms, 3), args.nt)

        tj_ma_ade, tj_ma_fde = compute_ade_fde(batch_cuda["ego_traj"][..., :4], dense_trajs[..., :-1, :4], tj_stl_input["dense_valids"])

        md.update("tj_acc", tj_acc.item())
        md.update("tj_scene_acc", tj_scene_acc.item())
        md.update("tj_std", tj_ma_std.item())
        md.update("tj_vol", tj_ma_vol.item())
        md.update("tj_ade", tj_ma_ade.item())
        md.update("tj_fde", tj_ma_fde.item())

        if args.extra_diversity:
            results = napi.measure_extra_diversity(
                dense_trajs[:, :-1].reshape(bs, args.n_randoms, 3, args.nt*4), 
                tj_scores.reshape(bs, args.n_randoms, 3), 
                tj_stl_input["dense_valids"].reshape(bs, args.n_randoms, 3), args.nt,
                dense_controls.reshape(bs, args.n_randoms, 3, args.nt*2),
                -args.mul_w_max, args.mul_w_max, -args.mul_a_max, args.mul_a_max,
            )
            for key in results:
                md.update("tj_"+key, results[key].item())

        tttt1=time.time()
        # neural net trajectories
        feature = None
        new_batch = {
            "ego_traj": batch_cuda["ego_traj"],
            "neighbors": batch_cuda["neighbors"],
            "currlane_wpts": batch_cuda["currlane_wpts"],
            "leftlane_wpts": batch_cuda["leftlane_wpts"],
            "rightlane_wpts": batch_cuda["rightlane_wpts"],
            "curr_id": batch_cuda["curr_id"],
            "left_id": batch_cuda["left_id"],
            "right_id": batch_cuda["right_id"],
            "neighbor_trajs_aug": batch_cuda["neighbors_traj"][..., :7],
            "gt_high_level": batch_cuda["gt_high_level"],
            "pre_stlp": batch_cuda["pre_stlp"],
        }
        new_batch = augment_batch_data(new_batch, gt_stlp, args, n_randoms=args.sampling_size)
        highlevel_new = new_batch["highlevel_dense"]
        states_new = states.unsqueeze(1).unsqueeze(1).repeat(1, args.sampling_size, 3, 1)
        states_flat_new = states_new.reshape(N, states_new.shape[-1])

        if args.diffusion:
            noise = torch.normal(0, 1, (N, args.nt * 2)).cuda().float()
            guidance_extras = (new_batch, states_flat_new, stls_cac) if args.guidance else None
            if args.time_profile:
                myt.add("start_diffusion")
            res = diffusion_rollout(noise, net, new_batch, highlevel_new, feature, args, coeffs, fastforward=False, n_randoms=args.sampling_size, return_feature=True, guidance_extras=guidance_extras)
            if args.time_profile:
                myt.add("end_diffusion")
            if args.diff_full:
                nn_controls, feature, nn_controls_list = res
            else:
                nn_controls, feature = res
                nn_controls_list=None
            
            if args.rect_head and args.not_use_rect==False:
                if args.multi_cands is not None:
                    # cat in the first dim (multi_cands, N, ...)
                    states_mul = states_flat_new.repeat(args.multi_cands, 1)
                    nn_ctrls_mul = torch.cat(nn_controls_list[-args.multi_cands:], dim=0)
                    nn_trajs_mul = generate_trajs(states_mul, nn_ctrls_mul, args.dt)
                    prev_stl_input = pre_prepare_stl_cache(new_batch, dense_trajs=nn_trajs_mul[:, :-1], repeat_n=args.multi_cands)
                    _, scores_hist_list, prev_acc = compute_stl_dense(prev_stl_input, stls_cac, 
                                        new_batch["highlevel_dense"].repeat((args.multi_cands, *[1]*(new_batch["highlevel_dense"].dim()-1))), 
                                        prev_stl_input["dense_valids"].reshape(-1), args)
                    # print(nn_ctrls_mul.shape, scores_hist_list.shape)
                    controls_hist_list = nn_ctrls_mul.reshape(args.multi_cands, nn_controls_list[-1].shape[0], args.nt, 2)
                    scores_hist_list = scores_hist_list.reshape(args.multi_cands, nn_controls_list[-1].shape[0])

                    scores_hist_max, scores_hist_max_i = torch.max(scores_hist_list, dim=0)
                    controls_hist_max = controls_hist_list[scores_hist_max_i, range(scores_hist_max_i.shape[0])]
                    
                    if args.time_profile:
                        myt.add("selected_stl_max")
                    
                    nn_controls = controls_hist_max
                    prev_scores = scores_hist_max
                    if args.no_refinenet:
                        nn_controls = nn_controls
                    else:
                        nn_controls = net.rect_forward(feature, highlevel_new, new_batch["stlp_dense"][:,0], nn_controls.detach(), prev_scores.detach(), extras=nn_controls_list)
                    if args.time_profile:
                        myt.add("rect_forward()")
                else:
                    prev_trajs = generate_trajs(states_flat_new, nn_controls, args.dt)
                    prev_stl_input = pre_prepare_stl_cache(new_batch, dense_trajs=prev_trajs[:,:-1])
                    _, prev_scores, prev_acc = compute_stl_dense(prev_stl_input, stls_cac, new_batch["highlevel_dense"], prev_stl_input["dense_valids"], args)
                    nn_controls = net.rect_forward(feature, highlevel_new, new_batch["stlp_dense"][:,0], nn_controls.detach(), prev_scores.detach(), extras=nn_controls_list)
                
                if args.n_rolls is not None:
                    for _ in range(args.n_rolls):
                        prev_trajs_re = generate_trajs(states_flat_new, nn_controls.detach(), args.dt)
                        prev_stl_input_re = pre_prepare_stl_cache(new_batch, dense_trajs=prev_trajs_re[:,:-1].detach())
                        _, prev_scores_re, prev_acc_re = compute_stl_dense(prev_stl_input_re, stls_cac, new_batch["highlevel_dense"], prev_stl_input_re["dense_valids"], args)
                        nn_controls = net.rect_forward(feature, highlevel_new, new_batch["stlp_dense"][:,0], nn_controls.detach(), prev_scores_re.detach(), extras=nn_controls_list)

                # further gradient
                if args.refinement:
                    K = 8  # 10, 8, 6, 4
                    N_ITERS = 50
                    STL_THRES = 0.0005
                    LR = 3e-1 #1e-1
                    lamdas = torch.ones(bs * args.sampling_size * 3, K).cuda().requires_grad_()
                    optimizer = torch.optim.Adam([lamdas], lr=LR)

                    nn_trajs = generate_trajs(states_flat_new, nn_controls, args.dt).reshape(N, args.nt+1, 4)
                    stl_input = pre_prepare_stl_cache(new_batch, dense_trajs=nn_trajs[:, :-1])
                    scores_list, scores, acc = compute_stl_dense(stl_input, stls_cac, new_batch["highlevel_dense"], stl_input["dense_valids"], args)

                    violated = (torch.logical_and(scores<=0, new_batch["valids_dense"].reshape(-1, )>0)).float()
                    violated = violated.reshape(bs * args.sampling_size * 3, 1, 1)

                    print("before, acc=%.3f"%(acc))
                    for opt_i in range(N_ITERS):
                        ratios = torch.softmax(lamdas, dim=-1)
                        # ratios = lamdas / torch.clip(torch.sum(lamdas, dim=-1, keepdim=True), 1e-5)
                        k_d_list={
                            2:[0], 3:[80, 95], 4:[80, 90, 95], 6:[0,50,80,90,95], 8:[0,50,80,85,90,95,98], 10:[0,50,80,85,90,95,96,97,98],
                            20:[0, 10, 30, 50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
                        }
                        combination = [nn_controls_list[i_index].detach() * ratios[..., i_i+1:i_i+2, None] for i_i, i_index in enumerate(k_d_list[K])]
                        optim_controls = nn_controls.detach() * ratios[..., 0:1, None] + torch.sum(torch.stack(combination, dim=-1), dim=-1)
                        
                        optim_controls = nn_controls.detach() * (1-violated.detach()) + violated.detach() * optim_controls
                        optim_trajs = generate_trajs(states_flat_new.detach(), optim_controls, args.dt)
                        optim_stl_input = pre_prepare_stl_cache(new_batch, dense_trajs=optim_trajs[:, :-1], detach=True)
                        
                        _, optim_scores, optim_acc = compute_stl_dense(optim_stl_input, stls_cac, new_batch["highlevel_dense"].detach(), optim_stl_input["dense_valids"].detach(), args)

                        loss = mask_mean(torch.nn.ReLU()(STL_THRES-optim_scores), new_batch["valids_dense"].reshape(-1, ).detach())

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    nn_controls = optim_controls.detach()

        elif args.vae:
            if args.use_init_hint:
                rand_w0 = uniform(-args.mul_w_max, args.mul_w_max, (bs, args.sampling_size, 3, args.nt)) * 0.1
                rand_a0 = uniform(-args.mul_a_max, args.mul_a_max, (bs, args.sampling_size, 3, args.nt))
                new_hint = torch.stack([rand_w0, rand_a0], dim=-1).cuda() # TODO
                if args.replace_hint:
                    new_hint[:, :args.n_randoms] = batch_cuda["params_init"].reshape(bs, args.n_randoms, 3, args.nt, 2)
                new_hint = new_hint.reshape(N, args.nt * 2)
                new_batch["params_init"] = new_hint
                # replace first 64 to the existing ones
            ext = {"highlevel": highlevel_new}
            gaussian_sample = torch.normal(0, 1, (bs * args.sampling_size * 3, args.vae_dim)).to(highlevel_new.device).float()
            results = net(new_batch, ext=ext, n_randoms=args.sampling_size, sample=gaussian_sample)
            nn_controls = results[0]

        elif args.bc:
            if args.use_init_hint:
                rand_w0 = uniform(-args.mul_w_max, args.mul_w_max, (bs, args.sampling_size, 3, args.nt)) * 0.1
                rand_a0 = uniform(-args.mul_a_max, args.mul_a_max, (bs, args.sampling_size, 3, args.nt))
                new_hint = torch.stack([rand_w0, rand_a0], dim=-1).cuda() # TODO
                if args.replace_hint:
                    new_hint[:, :args.n_randoms] = batch_cuda["params_init"].reshape(bs, args.n_randoms, 3, args.nt, 2)
                new_hint = new_hint.reshape(N, args.nt * 2)
                new_batch["params_init"] = new_hint
                # replace first 64 to the existing ones
            ext = {"highlevel": highlevel_new}
            nn_controls = net(new_batch, ext=ext, n_randoms=args.sampling_size)

        nn_trajs = generate_trajs(states_flat_new, nn_controls, args.dt).reshape(N, args.nt+1, 4)
        stl_input = pre_prepare_stl_cache(new_batch, dense_trajs=nn_trajs[:, :-1])
        scores_list, scores, acc, scene_acc = compute_stl_dense(stl_input, stls_cac, new_batch["highlevel_dense"], stl_input["dense_valids"], args, tj_scores=tj_scores, scene=True)
        
        tttt2=time.time()
        
        ma_std, ma_vol, ma_std_list, ma_vol_list = napi.measure_diversity(
            nn_trajs[:, :-1, :2].reshape(bs, args.sampling_size, 3, args.nt*2), 
            scores.reshape(bs, args.sampling_size, 3), 
            stl_input["dense_valids"].reshape(bs, args.sampling_size, 3), args.nt)
        
        ma_ade, ma_fde = compute_ade_fde(batch_cuda["ego_traj"][..., :4], nn_trajs[..., :-1, :4], stl_input["dense_valids"])

        md.update("acc", acc.item())
        md.update("scene_acc", scene_acc.item())
        md.update("ade", ma_ade.item())
        md.update("fde", ma_fde.item())
        md.update("std", ma_std.item())
        md.update("vol", ma_vol.item())
        md.update("time", tttt2-tttt1)

        if args.extra_diversity:
            results = napi.measure_extra_diversity(
                nn_trajs[:, :-1].reshape(bs, args.sampling_size, 3, args.nt*4), 
                scores.reshape(bs, args.sampling_size, 3), 
                stl_input["dense_valids"].reshape(bs, args.sampling_size, 3), args.nt,
                nn_controls.reshape(bs, args.sampling_size, 3, args.nt*2),
                -args.mul_w_max, args.mul_w_max, -args.mul_a_max, args.mul_a_max,
            )
            for key in results:
                md.update(key, results[key].item())    

        if args.extra_diversity:
            print("###[%02d] TJ acc:%.3f scene_acc:%.3f ade:%.3f fde:%.3f std:%.3f vol:%.3f area:%.3f s:%.3f u:%.3f| NN acc:%.3f scene_acc:%.3f ade:%.3f fde:%.3f std:%.3f vol:%.3f area:%.3f s:%.3f u:%.3f ||| T:%.3f"%(
                bi, md("tj_acc"), md("tj_scene_acc"), md("tj_ade"), md("tj_fde"), md("tj_std"), md("tj_vol"), md("tj_area"), md("tj_ent_s"), md("tj_ent_wa"),
                md("acc"), md("scene_acc"),  md("ade"), md("fde"), md("std"), md("vol"), md("area"), md("ent_s"), md("ent_wa"), md("time"),
            ))
        else:
            print("batch:%d tj_acc:%.3f(%.3f) acc:%.3f(%.3f) |  tj_std:%.3f(%.3f) tj_vol:%.3f(%.3f)  std:%.3f(%.3f)  vol:%.3f(%.3f)"%(
                bi, md["tj_acc"], md("tj_acc"), md["acc"], md("acc"), 
                md["tj_std"], md("tj_std"), md["tj_vol"], md("tj_vol"), 
                md["std"], md("std"), md["vol"], md("vol"), 
            ))

        # visualization
        if args.skip_nusc_load:
            batch_np = to_np_dict(batch)
            plot_debug_scene(batch_np, to_np(dense_trajs).reshape(bs, args.n_randoms, 3, args.nt+1, 4), to_np(tj_scores), 
                        to_np(nn_trajs.reshape(bs, args.sampling_size, 3, args.nt+1, 4)), to_np(scores), 
                        args, 0, i=0, tj_n_randoms=args.n_randoms, nn_n_randoms=args.sampling_size)
        else:
            if nusc is None:
                print("Wait for nuscene loading complete...")
                thread_nusc.join()
                nusc, nusc_map_d = result_queue.get()
                print("Loading nuscene was finished!")
                
            batch_np = to_np_dict(batch_cuda)
            for ego_only, opt_only in [[True, False], [False, True]]:
                if args.ego:
                    if ego_only==True:
                        ii_list = []
                        dvalid=stl_input["dense_valids"].reshape(bs, args.sampling_size, 3)
                        for ii_ in range(bs):
                            if mask_mean((scores.reshape(bs, args.sampling_size, 3)[ii_]>0).float(), dvalid[ii_]) > 0.75 and torch.sum(dvalid[ii_, 0, :])>1:
                                ii_list.append(ii_)
                        print(ii_list)
                elif args.other:
                    ii_list=[]
                    for ii_ in range(bs):
                        if (int(batch_np["traj_i"][ii_].item()), int(batch_np["ti"][ii_].item())) in interest_list:
                            ii_list.append(ii_)
                    if ego_only:
                        print(ii_list)
                else:
                    ii_list = [0]
                for ii_ in ii_list:
                    plot_paper_scene(nusc, nusc_map_d, data_loader.dataset.meta_d, batch_np, to_np(dense_trajs).reshape(bs, args.n_randoms, 3, args.nt+1, 4), to_np(tj_scores), 
                        to_np(nn_trajs.reshape(bs, args.sampling_size, 3, args.nt+1, 4)), to_np(scores), 
                        args, i=ii_, tj_n_randoms=args.n_randoms, nn_n_randoms=args.sampling_size, ego_only=ego_only, opt_only=opt_only)

    if args.time_profile:
        myt.print_profile()

def main():
    global args
    args = utils.setup_exp_and_logger(args, test=args.test)
    train_loader, val_loader, nusc, nusc_map_d, meta_list = get_dataloader(args)

    # tasks that not involving training/neural nets
    if args.collect_data:  # data collection -> cache
        collect_nuscene_data(train_loader, meta_list)
        return

    # TODO(trajopt should also be here?)
    # build the STL formula
    stls_cac = build_stl_cache(args)

    if args.check_stl_params:
        check_stl_params(train_loader, meta_list, stls_cac, args)
        return

    # async load the nuscene data
    if all([not args.skip_nusc_load, nusc is None]):
        result_queue = Queue()
        thread_nusc = threading.Thread(target=napi.get_nuscenes, args=(args.mini, result_queue))
        thread_nusc.start()
    else:
        result_queue = None
        thread_nusc = None

    # construct and load the neural network
    net = Net(args).cuda()
    if args.net_pretrained_path is not None:
        net.load_state_dict(torch.load(utils.get_model_path(args.net_pretrained_path)), strict=(not args.rect_head))
    
    multi_check = any([args.diffusion, args.vae, args.bc]) and args.gt_data_training==False

    eta = utils.EtaEstimator(
        0, args.epochs * (len(train_loader)+len(val_loader)), 1, viz_freq=args.viz_freq,
        epochs=args.epochs, total_train_bs=len(train_loader.dataset), total_val_bs=len(val_loader.dataset), batch_size=args.batch_size)
    
    coeffs = get_diffusion_coeffs(args)

    if args.run_sampling_test:
        run_sampling_test(stls_cac, val_loader, net, coeffs, args, result_queue, thread_nusc)
        return

    if args.rect_head:
        if args.joint:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(net.rect_net.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    if args.trajopt_only:
        scores_cache_list = []
    else:
        scores_cache_list = None

    pre_comp = {"train":{}, "val":{}}
    dataloader_d = {"train": train_loader, "val": val_loader}

    for epi in range(args.epochs):
        viz_cache = {}
        for mode in ["train", "val"]:            
            all_scores_list = {}
            met_d = utils.MeterDict()
            data_loader = dataloader_d[mode]
            for bi, batch in enumerate(data_loader):           
                ttt1 = time.time()

                batch_cuda = dict_to_cuda(batch)

                gt_trajs = batch_cuda["ego_traj"][..., :4]
                states = gt_trajs[..., 0, :4]
                bs = states.shape[0]
                
                ttt2 = time.time()
                # TODO whether we should do this outside/inside dataloader?
                if args.gt_nei:
                    batch_cuda["neighbor_trajs_aug"] = batch_cuda["neighbors_traj"][..., :7]
                else:
                    batch_cuda["neighbor_trajs_aug"] = get_neighbor_trajs(batch_cuda["neighbors"], args.nt, args.dt)
                gt_stlp = infer_gt_stlp(batch_cuda, gt_trajs, args)  # (bs, 6)

                ttt3 = time.time()
                if multi_check:
                    batch_cuda = augment_batch_data(batch_cuda, gt_stlp, args)

                    # ego (N, nrand, 3, 6)
                    n = bs * args.n_randoms * 3
                    dense_states = states.unsqueeze(1).unsqueeze(1).repeat(1, args.n_randoms, 3, 1)
                    dense_states_flat = dense_states.reshape(n, dense_states.shape[-1])
                    highlevel_dense = batch_cuda["highlevel_dense"]
                    avg_speed_gt = torch.mean(batch_cuda["ego_traj"][..., 3])
                    
                    if args.load_tj:
                        dense_controls = batch_cuda["params"]
                        dense_trajs = generate_trajs(dense_states, dense_controls, args.dt)
                        dense_scores = batch_cuda["tj_scores_prior"].reshape(bs*args.n_randoms, 3)
                        dense_valids = batch_cuda["valids_dense"]
                        all_scores_trajopt = evaluate_all_scores(dense_scores, batch_cuda["gt_high_level"], dense_valids)
                        avg_acc = torch.mean((dense_scores>=0).float() * dense_valids) / torch.clip(torch.mean(dense_valids), 1e-3)
                        tj_rd = {}
                        if args.measure_diversity:
                            if epi==0:
                                ma_std, ma_vol, ma_std_list, ma_vol_list = napi.measure_diversity(
                                    dense_trajs[..., :-1, :2].reshape(bs, args.n_randoms, 3, args.nt*2), 
                                    dense_scores.reshape(bs, args.n_randoms, 3), 
                                    dense_valids.reshape(bs, args.n_randoms, 3), args.nt)
                                tj_rd["ma_std"] = torch.tensor([ma_std]).float().to(dense_trajs.device)
                                tj_rd["ma_vol"] = torch.tensor([ma_vol]).float().to(dense_trajs.device)
                                tj_rd["ma_std_list"] = torch.from_numpy(np.stack(ma_std_list, axis=-1)).float().to(dense_trajs.device)
                                tj_rd["ma_vol_list"] = torch.from_numpy(np.stack(ma_vol_list, axis=-1)).float().to(dense_trajs.device)
                            else:
                                tj_rd["ma_std"] = pre_comp[mode]["ma_std"]
                                tj_rd["ma_vol"] = pre_comp[mode]["ma_vol"]
                        scene_acc = torch.mean((torch.max(dense_scores.reshape(bs, args.n_randoms, 3), dim=1)[0]>=0).float() * (dense_valids.reshape(bs, args.n_randoms, 3)[:,0]), dim=0) / torch.clip(torch.mean(dense_valids.reshape(bs, args.n_randoms, 3)[:,0], dim=0), 1e-3)

                    else:
                        ###########################
                        # TRAJECTORY OPTIMIZATION #
                        # ctrls (N, nrand, 3, nt, 2)
                        if epi==0:
                            save_trajopt_params(batch_cuda["params"], "init", batch_cuda["traj_i"], batch_cuda["ti"], args, save_stlp=batch_cuda["stlp_dense"])
                        
                        if epi < args.opt_epochs:
                            dense_controls = batch_cuda["params"] = batch_cuda["params"].requires_grad_()
                            traj_optimizer = torch.optim.Adam([batch_cuda["params"]], lr=args.trajopt_lr)
                        else:
                            dense_controls = batch_cuda["params"] = batch_cuda["params"].requires_grad_()
                        opt_iters = args.traj_opt_iters if epi < args.opt_epochs else 1
                        for ii in range(opt_iters):
                            dense_trajs = generate_trajs(dense_states, dense_controls, args.dt)
                            stl_input_cache = pre_prepare_stl_cache(batch_cuda)
                            trajopt_loss, dense_loss, reg_loss, avg_acc, avg_speed, dense_scores, dense_valids, all_scores_trajopt, tj_rd, scene_acc = compute_trajopt_loss_lite(dense_controls, dense_trajs, stls_cac, stl_input_cache, ii, opt_iters)
                            if epi < args.opt_epochs:
                                traj_optimizer.zero_grad()
                                trajopt_loss.backward()
                                traj_optimizer.step()
                                if args.trajopt_only and (ii in [0,opt_iters//2,opt_iters-1] or ii % 1000==0):
                                    print(ii, dense_loss.item(), avg_acc.item(), scene_acc[0].item(), scene_acc[1].item(), scene_acc[2].item())
                        # save the params data
                        if epi < args.opt_epochs:
                            save_trajopt_params(batch_cuda["params"], "final", batch_cuda["traj_i"], batch_cuda["ti"], args)                    
                            save_trajopt_params(dense_scores.reshape(bs, args.n_randoms, 3), "scores", batch_cuda["traj_i"], batch_cuda["ti"], args)

                        if scores_cache_list is not None:
                            sc = (dense_scores.reshape(bs, args.n_randoms, 3)>0).float()
                            dv = dense_valids.reshape(bs, args.n_randoms, 3)
                            lb = batch_cuda["gt_high_level"]
                            lb_ = batch_cuda["gt_high_level"][..., None]
                            lb_sc = sc[..., 0] * ((lb-1)*(lb-2)!=0).float() + sc[..., 1] * (lb==1).float() + sc[..., 2] * (lb==2).float()
                            lb_dv = dv[..., 0] * ((lb-1)*(lb-2)!=0).float() + dv[..., 1] * (lb==1).float() + dv[..., 2] * (lb==2).float()
                            olb_sc = sc[..., 1:3] * ((lb_-1)*(lb_-2)!=0).float() + sc[..., 0:2] * (lb_==2).float() + sc[..., [0, 2]] * (lb_==1).float()
                            olb_dv = dv[..., 1:3] * ((lb_-1)*(lb_-2)!=0).float() + dv[..., 0:2] * (lb_==2).float() + dv[..., [0, 2]] * (lb_==1).float()
                            olb_sc = olb_sc.reshape(bs, args.n_randoms * 2)
                            olb_dv = olb_dv.reshape(bs, args.n_randoms * 2)
                            scores_cache_list.append(
                                [batch_cuda["traj_i"], batch_cuda["ti"], batch_cuda["gt_high_level"][:,0], 
                                mask_mean(sc.reshape(bs, -1), dv.reshape(bs, -1), dim=-1),  # the whole accuracy
                                mask_mean(sc[..., 0], dv[..., 0], dim=-1),  # the action=0 accuracy
                                mask_mean(sc[..., 1], dv[..., 1], dim=-1),  # the action=1 accuracy
                                mask_mean(sc[..., 2], dv[..., 2], dim=-1),  # the action=2 accuracy
                                mask_mean(lb_sc, lb_dv, dim=-1),  # the in-label accuracy
                                mask_mean(olb_sc, olb_dv, dim=-1),  # the out-label accuracy
                                ])
                    
                    met_d.update("avg_acc", avg_acc.item())
                    if args.trajopt_only:
                        met_d.update("trajopt_loss", trajopt_loss.item())
                        met_d.update("dense_loss", dense_loss.item())
                        met_d.update("reg_loss", reg_loss.item())
                        met_d.update("avg_speed", avg_speed.item())
                        met_d.update("avg_speed_gt", avg_speed_gt.item())
                    
                    ttt4 = time.time()
                    rect_trajs = None
                    nn_controls_adj = None
                    nn_controls_list_adj = None
                    if not args.trajopt_only: # TODO diffusion loss and neural policy input designs
                        diffusion_extras = vae_extras = bc_extras = None
                        if args.diffusion:
                            noise, steps, noised_cmds_a, noised_cmds_b = diffusion_prep(dense_controls, n_randoms=args.n_randoms, coeffs=coeffs)
                            ext = {"timestep":steps , "highlevel": highlevel_dense, "noise": noised_cmds_b}
                            est_cmds_a, feature = net(batch_cuda, ext=ext, get_feature=True)
                            est_cmds_a = est_cmds_a.reshape(n, args.nt*2)
                            if args.grad_rollout or args.rect_head:
                                guidance_extras = (batch_cuda, dense_states_flat, stls_cac) if args.guidance else None
                                resres = diffusion_rollout(noise, net, batch_cuda, highlevel_dense, feature, args, coeffs, fastforward=False, guidance_extras=guidance_extras)
                                if args.diff_full:
                                    nn_controls, nn_controls_list = resres
                                else:
                                    nn_controls = resres
                                    nn_controls_list = None
                            else:
                                nn_controls = diffusion_rollout(noise, net, batch_cuda, highlevel_dense, feature, args, coeffs, fastforward=(epi%args.viz_freq!=0 and epi!=args.epochs-1))
                            
                            if args.rect_head:
                                if args.multi_cands is not None:
                                    # cat in the first dim (multi_cands, N, ...)
                                    states_mul = dense_states_flat.repeat(args.multi_cands, 1)
                                    nn_ctrls_mul = torch.cat(nn_controls_list[-args.multi_cands:], dim=0)
                                    nn_trajs_mul = generate_trajs(states_mul, nn_ctrls_mul, args.dt)
                                    prev_stl_input = pre_prepare_stl_cache(batch_cuda, dense_trajs=nn_trajs_mul[:, :-1], repeat_n=args.multi_cands)
                                    _, scores_hist_list, prev_acc = compute_stl_dense(prev_stl_input, stls_cac, 
                                                        batch_cuda["highlevel_dense"].repeat((args.multi_cands, *[1]*(batch_cuda["highlevel_dense"].dim()-1))), 
                                                        prev_stl_input["dense_valids"].reshape(-1), args)

                                    controls_hist_list = nn_ctrls_mul.reshape(args.multi_cands, nn_controls_list[-1].shape[0], args.nt, 2)
                                    scores_hist_list = scores_hist_list.reshape(args.multi_cands, nn_controls_list[-1].shape[0])

                                    scores_hist_max, scores_hist_max_i = torch.max(scores_hist_list, dim=0)
                                    controls_hist_max = controls_hist_list[scores_hist_max_i, range(scores_hist_max_i.shape[0])]
                                    
                                    nn_controls = controls_hist_max
                                    prev_scores = scores_hist_max
                                    rect_controls = net.rect_forward(feature, highlevel_dense, batch_cuda["stlp_dense"][:,0], nn_controls.detach(), prev_scores.detach(), extras=nn_controls_list)
                                else:
                                    prev_trajs = generate_trajs(dense_states_flat, nn_controls, args.dt)
                                    prev_stl_input = pre_prepare_stl_cache(batch_cuda, dense_trajs=prev_trajs[:, :-1])
                                    _, prev_scores, _ = compute_stl_dense(prev_stl_input, stls_cac, batch_cuda["highlevel_dense"], prev_stl_input["dense_valids"].reshape(-1), args)
                                    rect_controls = net.rect_forward(feature, highlevel_dense, batch_cuda["stlp_dense"][:,0], nn_controls.detach(), prev_scores.detach(), extras=nn_controls_list)
                            else:
                                rect_controls = None
                            diffusion_extras=(noised_cmds_a, est_cmds_a, highlevel_dense, dense_scores, dense_valids, epi, noise, nn_controls, steps, rect_controls)
                        elif args.bc:
                            ext = {"highlevel": highlevel_dense, }
                            nn_controls = net(batch_cuda, ext=ext)
                            bc_extras = (nn_controls, dense_controls, dense_trajs, highlevel_dense, dense_scores, dense_valids, epi)
                        elif args.vae:
                            noise = torch.normal(0, 1, (n, args.vae_dim)).cuda().float()
                            ext = {"highlevel": highlevel_dense, "noise": noise, "trajopt_controls": dense_controls}
                            nn_controls, latent_mean, latent_logstd, latent_std = net(batch_cuda, ext=ext)
                            vae_extras=(nn_controls, dense_controls, dense_trajs, highlevel_dense, dense_scores, dense_valids, (latent_mean, latent_logstd, latent_std), epi)                 

                        nn_trajs = generate_trajs(dense_states_flat, nn_controls, args.dt)  # (n, nt, 6?)
                        if args.rect_head:
                            rect_trajs = generate_trajs(dense_states_flat, rect_controls, args.dt)  # (n, nt, 6?)

                        rd, all_scores = compute_policy_loss(batch_cuda, None, stls_cac, nn_trajs, rect_trajs, dense_trajs, args, 
                                    diffusion_extras=diffusion_extras, vae_extras=vae_extras, bc_extras=bc_extras,
                                    nn_controls_adj=nn_controls_adj, nn_controls_list_adj=nn_controls_list_adj,
                                    opt_controls=dense_controls,
                                    )

                    the_all_scores = all_scores_trajopt if args.trajopt_only else all_scores 
                    for key in the_all_scores:
                        if key not in all_scores_list:
                            all_scores_list[key] = []
                        all_scores_list[key] += the_all_scores[key]
                    ttt5 = time.time()
                else:
                    # for VAE/diffusion for gt data 
                    bs = batch_cuda["gt_high_level"].shape[0]        
                    gt_controls = (batch_cuda["ego_traj"][:, 1:,2:4] - batch_cuda["ego_traj"][:, :-1,2:4]) / args.dt
                    gt_controls = torch.cat([gt_controls, gt_controls[:, -1:, ]], dim=1)
                    states_mul = states[:, None, :].repeat(1, args.n_randoms, 1).reshape(bs * args.n_randoms, 4)
                    if args.diffusion:
                        noise, steps, noised_cmds_a, noised_cmds_b = diffusion_prep(gt_controls, n_randoms=args.n_randoms, coeffs=coeffs, mono=True)
                        ext = {"timestep":steps , "highlevel": batch_cuda["gt_high_level"], "noise": noised_cmds_b, "gt_stlp": gt_stlp}
                        est_cmds_a, feature = net(batch_cuda, ext=ext, get_feature=True)
                        est_cmds_a = est_cmds_a.reshape(-1, args.nt*2)
                        nn_controls_mul = diffusion_rollout(noise, net, batch_cuda, batch_cuda["gt_high_level"], 
                                                        feature, args, coeffs, fastforward=(epi%args.viz_freq!=0 and epi!=args.epochs-1), mono=True, tmp_stlp=gt_stlp)
                    elif args.vae:
                        noise = torch.normal(0, 1, (bs * args.n_randoms, args.vae_dim)).cuda().float()
                        ext = {"gt_stlp": gt_stlp, "highlevel": batch_cuda["gt_high_level"], "gt_controls": gt_controls, "noise":noise}
                        nn_controls_mul, latent_mean, latent_logstd, latent_std = net(batch_cuda, ext=ext)
                    else:
                        raise NotImplementedError
                    
                    nn_trajs_mul_flat = generate_trajs(states_mul, nn_controls_mul, args.dt)
                    nn_trajs_mul = nn_trajs_mul_flat.reshape(bs,args.n_randoms,args.nt+1,4)
                    nn_trajs = nn_trajs_mul

                    l2_loss = torch.mean(torch.mean(torch.square(nn_controls_mul.reshape(bs, args.n_randoms, args.nt, 2) - gt_controls[:, None]), dim=-1),dim=-1)

                    min_val, min_idx = torch.min(l2_loss, dim=1)
                    
                    # mono_trajs = nn_trajs_mul[range(bs), min_idx]
                    mono_stl_input = pre_prepare_stl_cache(batch_cuda, dense_trajs=nn_trajs_mul_flat[:, :-1], mono=True, mono_n=args.n_randoms, gt_stlp=gt_stlp)
                    _, mono_scores, mono_acc = compute_stl_dense(mono_stl_input, stls_cac, mono_stl_input["gt_high_level"], mono_stl_input["dense_valids"].reshape(-1), args)

                    # gt scores
                    gt_stl_input = pre_prepare_stl_cache(batch_cuda, dense_trajs=batch_cuda["ego_traj"], mono=True, mono_n=1, gt_stlp=gt_stlp)
                    _, scores_gt_all, acc_gt = compute_stl_dense(gt_stl_input, stls_cac, gt_stl_input["gt_high_level"], gt_stl_input["dense_valids"].reshape(-1), args)

                    # minimum-over-n loss
                    if args.diffusion:
                        loss_diffusion = torch.mean(torch.square(noise-est_cmds_a))
                        loss_vae_bc = loss_diffusion * 0
                        loss_vae_kl = loss_diffusion * 0
                    elif args.bc:
                        loss_vae_bc = torch.mean(min_val) * args.bc_weight
                        loss_vae_kl = (-1/2 * torch.mean(1 + 2*latent_logstd - latent_mean*latent_mean - latent_std*latent_std)) * args.weight_vae_kl
                    loss_stl = torch.mean(torch.nn.ReLU()(args.stl_nn_thres - mono_scores)) * args.stl_weight
                    loss = loss_diffusion + loss_vae_bc + loss_vae_kl + loss_stl

                    scores_all = dense_scores = mono_scores
                    dense_valids = torch.ones_like(mono_scores)

                    rd = {"loss_diffusion": loss_diffusion,
                        "loss_vae_bc":loss_vae_bc, "loss_vae_kl":loss_vae_kl, "loss_stl":loss_stl, "loss":loss, "acc":mono_acc, "acc_gt": acc_gt,
                        "avg_speed":torch.mean(nn_trajs_mul_flat[..., 3]), 
                        "avg_speed_gt": torch.mean(batch_cuda["ego_traj"][..., 3]),
                        "scores_all": scores_all,
                        "scores_gt_all": scores_gt_all,
                    }
                    tj_rd={}

                if args.trajopt_only==False:
                    if args.rect_head:
                        self_trajs = rect_trajs
                    else:
                        self_trajs = nn_trajs

                eta.smart_update(epi, duration=time.time()-ttt1, bs=batch["ego_traj"].shape[0], mode=mode, bi=bi)

                for key in ["traj_i", "ti", "ego_traj", "gt_high_level", "neighbors", "currlane_wpts", "leftlane_wpts", "rightlane_wpts"]:
                    add_to(viz_cache, mode, key, batch_cuda[key])
                add_to(viz_cache, mode, "gt_stlp", gt_stlp)
                add_to(viz_cache, mode, "dense_scores", dense_scores)
                add_to(viz_cache, mode, "dense_valids", dense_valids)

                if multi_check:
                    if bi < 10:
                        add_to(viz_cache, mode, "dense_trajs", dense_trajs)
                        if args.trajopt_only==False:
                            add_to(viz_cache, mode, "nn_trajs", self_trajs)
                else:
                    add_to(viz_cache, mode, "nn_trajs", self_trajs)

                if args.trajopt_only==False:
                    for key in ["scores_all", "scores_gt_all"]:
                        add_to(viz_cache, mode, key, rd[key])

                    ttt6 = time.time()

                    if mode=="train":
                        optimizer.zero_grad()
                        rd["loss"].backward()
                        optimizer.step()
                
                    ttt7 = time.time()

                    for key in ["loss", "loss_diffusion", "loss_vae_bc", "loss_vae_kl", "loss_diversity","loss_coll", "loss_reg", "loss_adj", "loss_cover",
                                "loss_bc", "loss_bc_masked", "loss_stl", "avg_speed", "avg_speed_gt", "acc", "acc_gt", "ma_std", "ma_vol"]:
                        if key in rd:
                            met_d.update(key, rd[key].item())        
                    for key in ["ma_std", "ma_vol"]:
                        if key in tj_rd:
                            met_d.update("tj_"+key, tj_rd[key].item())        
                    ttt8 = time.time()
                # print("%.3f | %.3f %.3f %.3f %.3f %.3f %.3f %.3f"%(ttt8-ttt1, ttt2-ttt1,ttt3-ttt2,ttt4-ttt3,ttt5-ttt4,ttt6-ttt5,ttt7-ttt6,ttt8-ttt7))

                if multi_check and epi==0 and bi==len(data_loader)-1 and args.measure_diversity:
                    pre_comp[mode]["ma_std"] = torch.tensor([met_d("tj_ma_std")]).float().cuda()
                    pre_comp[mode]["ma_vol"] = torch.tensor([met_d("tj_ma_vol")]).float().cuda()

                header_str = "%-5s[%03d %3d/%3d]"%(mode.capitalize(), epi, bi, len(data_loader))
                if epi % args.epi_print_freq == 0 and ((mode=="train" and bi % args.print_freq == 0) or bi == len(data_loader)-1):
                    if args.trajopt_only:
                        loss_str = print_for(met_d, {"loss":"trajopt_loss", "stl":"dense_loss", "reg":"reg_loss", "tacc":"avg_acc"}, just_avg=(mode=="val"))
                    elif args.diffusion:
                        loss_str = print_for(met_d, {"loss":"loss", "stl":"loss_stl", "dfsion":"loss_diffusion", "adj":"loss_adj","cov":"loss_cover","dvs":"loss_diversity", "reg":"loss_reg"}, just_avg=(mode=="val"))
                    elif args.bc:
                        loss_str = print_for(met_d, {"loss":"loss", "stl":"loss_stl", "bc":"loss_bc"}, just_avg=(mode=="val"))
                    elif args.vae:
                        loss_str = print_for(met_d, {"loss":"loss", "stl":"loss_stl", "vae_bc":"loss_vae_bc", "vae_kl":"loss_vae_kl", "cl":"loss_coll"}, just_avg=(mode=="val"))
                    else:
                        loss_str = print_for(met_d, {"loss":"loss", "stl":"loss_stl", "bc":"loss_bc"}, just_avg=(mode=="val"))
                    extra_str = ""
                    if multi_check and args.trajopt_only==False:
                        extra_str = print_for(met_d, {"tloss":"dense_loss", "tacc":"avg_acc"}, just_avg=(mode=="val"))
                    if args.trajopt_only:
                        meas_str = "avg_speed:(%.3f) |gt speed:(%.3f) T:%s ETA:%s"%(met_d("avg_speed"), met_d("avg_speed_gt"), 
                                                                                       eta.elapsed_str(), eta.eta_str_smart())
                    else:
                        meas_str = "avg_speed:(%.3f) |gt:(%.3f)  acc:(%.3f) |gt:(%.3f) T:%s ETA:%s"%(met_d("avg_speed"), met_d("avg_speed_gt"), met_d("acc"), met_d("acc_gt"),
                                                                                                        eta.elapsed_str(), eta.eta_str_smart())
                    if multi_check and args.measure_diversity:
                        diverse_str = " {%s}"%(print_for(met_d, {"std":"ma_std", "vol":"ma_vol", "tj_std":"tj_ma_std", "tj_vol":"tj_ma_vol"}, just_avg=(mode=="val")))
                    else:
                        diverse_str = ""
                    print("%s %s %s%s %s"%(header_str, loss_str, extra_str, diverse_str, meas_str))

                if bi==len(data_loader)-1 and multi_check:
                    for key in all_scores_list:
                        if len(all_scores_list[key])>0:
                            all_scores_list[key] = torch.cat(all_scores_list[key], dim=0)
                    print_all_scores(mode, all_scores_list)
                
        # save the network model
        utils.save_model_freq_last(net.state_dict(), args.model_dir, epi, args.save_freq, args.epochs)
        if args.trajopt_only:
            scores_cache_list = np.concatenate([np.stack([to_np(xxx) for xxx in xx], axis=-1) for xx in scores_cache_list], axis=0)
            np.savez("%s/trajopt_stat.npz"%(args.exp_dir_full), data=scores_cache_list)

        # numpy-cache
        for mode in viz_cache:
            for key in viz_cache[mode]:
                if len(viz_cache[mode][key])!=0:
                    viz_cache[mode][key] = to_np(torch.cat(viz_cache[mode][key], dim=0)) 

        # plot scores high low
        if args.trajopt_only==False:
            for mode in viz_cache:
                print("%-5s| gt score min:%.3f max:%.3f tj score min:%.3f max:%.3f nn score min:%.3f max:%.3f"%(
                    mode, 
                    np.min(viz_cache[mode]["scores_gt_all"]), np.max(viz_cache[mode]["scores_gt_all"]),
                    np.min(viz_cache[mode]["dense_scores"] * viz_cache[mode]["dense_valids"]), np.max(viz_cache[mode]["dense_scores"] * viz_cache[mode]["dense_valids"]),
                    np.min(viz_cache[mode]["scores_all"] * viz_cache[mode]["dense_valids"].flatten()), np.max(viz_cache[mode]["scores_all"] * viz_cache[mode]["dense_valids"].flatten()),
                ))

        if (epi % args.viz_freq == 0 or epi==args.epochs-1) and not args.no_viz:
            if args.skip_nusc_load==False:  # print for nuscene visulization
                if nusc is None:
                    print("Wait for nuscene loading complete...")
                    thread_nusc.join()
                    nusc, nusc_map_d = result_queue.get()
                    print("Loading nuscene was finished!")

                viz_t1 = time.time()
                for mode in ["train", "val"]:  # plot the first x trajs
                    existing_trajs = []
                    existing_iis = []
                    for ii in range(viz_cache[mode]["traj_i"].shape[0]):
                        traj_i = viz_cache[mode]["traj_i"][ii]
                        if traj_i not in existing_trajs:
                            existing_trajs.append(traj_i)
                            existing_iis.append(ii)
                    for k, ii in enumerate(existing_iis):
                        if k >= args.num_viz:
                            continue
                        if multi_check:
                            plot_nuscene_viz(ii, mode, viz_cache, epi, nusc, nusc_map_d, dataloader_d[mode], multi_check=multi_check, opt_only=True, args=args)
                            if args.trajopt_only==False:
                                plot_nuscene_viz(ii, mode, viz_cache, epi, nusc, nusc_map_d, dataloader_d[mode], multi_check=multi_check, ego_only=True, args=args)
                        else:
                            plot_nuscene_viz(ii, mode, viz_cache, epi, nusc, nusc_map_d, dataloader_d[mode], multi_check=multi_check, args=args)
            else:
                viz_t1 = time.time()
                dense_trajs = viz_cache["val"]["dense_trajs"]
                dense_scores = viz_cache["val"]["dense_scores"]
                self_trajs = viz_cache["val"]["nn_trajs"]
                scores_all = viz_cache["val"]["scores_all"]
                for select_i in [0,1,2,3,4]:
                    plot_debug_scene(viz_cache["val"], dense_trajs, dense_scores, self_trajs, scores_all, args, epi, i=select_i)
            eta.update_viz_time(time.time()-viz_t1) 


def generate_parser():
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--seed", type=int, default=1007)
    add("--exp_name", '-e', type=str, default=None)
    add("--gpus", type=str, default="0")
    add("--epochs", type=int, default=500)
    add("--test", action="store_true", default=False)
    add("--net_pretrained_path", "-P", type=str, default=None)
    add("--num_workers", type=int, default=8)
    add("--batch_size", "-b", type=int, default=128)
    add("--lr", type=float, default=3e-4)
    add("--hiddens", type=int, nargs="+", default=[256, 256])
    add("--print_freq", type=int, default=10)
    add("--save_freq", type=int, default=100)
    add("--viz_freq", type=int, default=50)
    add("--num_viz", type=int, default=10)

    add("--no_viz", action="store_true", default=False)
    add("--mini", action="store_true", default=False)
    add("--train_ratio", type=float, default=0.7)
    add("--n_neighbors", "-N", type=int, default=8)
    add("--n_randoms", type=int, default=64)
    add("--n_segs", type=int, default=15)
    add("--n_expands", type=int, default=4)
    add("--collect_data", action="store_true", default=False)
    add("--offline", action="store_true", default=False)
    add("--cache_path", type=str, default="e0_nusc_cache")
    
    add("--refined_safety", action="store_true", default=False)
    add("--ego_L", type=float, default=4.084)
    add("--ego_W", type=float, default=1.730)
    add("--refined_nL", type=int, default=4)
    add("--refined_nW", type=int, default=1)
    add("--nt", type=int, default=20)
    add("--dt", type=float, default=0.5)
    add("--mul_w_max", type=float, default=0.5)
    add("--mul_a_max", type=float, default=5.0)
    add("--smoothing_factor", type=float, default=100.0)

    add("--debug", action="store_true", default=False)
    add("--use_gt_stlp", action="store_true", default=False)
    add("--skip_nusc_load", action="store_true", default=False)
    add("--clip_dist", action="store_true", default=False)
    add("--anno_path", type=str, default="annotated_data_trainval")
    add("--gt_nei", action='store_true', default=False)
    
    add("--stl_bc_mask", action='store_true', default=False)
    add("--stl_nn_thres", type=float, default=0.0005)
    add("--stl_trajopt_thres", type=float, default=0.01)
    add("--trajopt_only", action="store_true", default=False)
    add("--traj_opt_iters", type=int, default=2000)
    add("--trajopt_lr", type=float, default=0.005)
    add("--opt_epochs", type=int, default=0)
    add("--trajopt_save_freq", type=int, default=1000)
    add("--params_load_path", "-P2", type=str, default="e1_nusc_trajopt")
    add("--inline", action="store_true", default=False)
    add("--use_init_hint", action="store_true", default=False)
    add("--generate_split_on_the_fly", action='store_true', default=False)
    add("--check_stl_params", action='store_true', default=False)
    add("--filter_traj", type=int, nargs="+", default=None)
    add("--norm_stl", action='store_true', default=False)
    add("--flex", action="store_true", default=False) # for better pstl estimation heur
    add("--load_stlp", action="store_true", default=False)
    add("--load_tj", action='store_true', default=False)
    
    add("--stl_weight", type=float, default=1.0)
    add("--bc", action="store_true", default=False)
    add("--bc_weight", type=float, default=0.0)
    add("--vae", action='store_true', default=False)
    add("--vae_dim", type=int, default=64)
    add("--weight_vae_bc", type=float, default=1.0)
    add("--weight_vae_kl", type=float, default=1.0)
    add("--diffusion", action="store_true", default=False)
    add("--diffusion_steps", type=int, default=100)
    add("--diffusion_weight", type=float, default=1.0)
    add("--beta_start", type=float, default=1e-4)
    add("--beta_end", type=float, default=0.02)
    add("--cos", action='store_true', default=False)
    add("--reg_loss", type=float, default=10.0)
    add("--grad_rollout", action='store_true', default=False)

    add("--rect_head", action="store_true", default=False)
    add("--rect_hiddens", type=int, nargs="+", default=[256, 256])
    add("--rect_reg_loss", type=float, default=0.000)
    add("--joint", action="store_true", default=False)
    add("--extra_rect_reg", type=float, default=0.0)
    add("--not_use_rect", action='store_true', default=False)

    add("--measure_diversity", action='store_true', default=False)
    add("--extra_diversity", action='store_true', default=False)
    add("--viz_correct", action='store_true', default=False)
    add("--epi_print_freq", type=int, default=1)

    add("--run_sampling_test", action='store_true', default=False)
    add("--sampling_size", type=int, default=64)
    add("--n_trials", type=int, default=100)  
    add("--replace_hint", action='store_true', default=False)
    
    add("--diff_full", action='store_true', default=False)
    add("--refinement", action='store_true', default=False)
    add("--raw_refinement", action='store_true', default=False)
    
    add("--diverse_loss", action='store_true', default=False)
    add("--diversity_weight", type=float, default=1.0)
    add("--diversity_scale", type=float, default=1.0)
    add("--no_arch", action='store_true', default=False)
    add("--n_shards", type=int, default=4)
    add("--diverse_fuse_type", type=str, default="add")
    add("--diverse_detach", action='store_true', default=False)
    
    # testing
    add("--test_t1", action='store_true', default=False)
    add("--test_scenes", action='store_true', default=False)
    add("--test_aggressive", action='store_true', default=False)
    add("--viz_last", action='store_true', default=False)
    add("--lite_refine", action='store_true', default=False)

    # advance rect_net
    add("--interval", action='store_true', default=False)
    add("--diffusion_clip", action='store_true', default=False)
    add("--multi_cands", type=int, default=None)

    # gt-data training
    add("--gt_data_training", action='store_true', default=False)
    add("--collision_loss", type=float, default=None)  # TraffcSim
    add("--guidance", action='store_true', default=False)    # CTG
    add("--guidance_niters", type=int, default=3)
    add("--guidance_before", type=int, default=1000)
    add("--guidance_lr", type=float, default=0.01)
    add("--guidance_reverse", action='store_true', default=False)
    add("--guidance_sets", nargs="+", type=int, default=None)
    add("--guidance_freq", type=int, default=None)

    add("--oracle_filter", action='store_true', default=False)
    add("--clip_rect", action='store_true', default=False)
    add("--ego", action='store_true', default=False)
    add("--other", action='store_true', default=False)
    
    add("--n_rolls", type=int, default=None)
    add("--suffix", type=str, default=None)
    add("--backup", action='store_true', default=False)
    add('--no_refinenet', action='store_true', default=False)
    add('--time_profile', action='store_true', default=False)
    args = parser.parse_args()
    args.gt_nei = True
    args.stl_bc_mask = True
    args.cos = True
    if args.collect_data==False and args.trajopt_only==False:
        args.measure_diversity = True
    if args.run_sampling_test:
        args.test = True
        args.extra_diversity=True
    if args.collect_data:
        args.epochs = 1
        args.batch_size = 1024
        args.viz_freq = 10
        args.print_freq = 1
        args.uturn = True
    if args.trajopt_only:
        args.opt_epochs = 1
        args.epochs = 1
        args.batch_size = 1024
        args.viz_freq = 10
        args.diffusion = True
        args.num_viz = 256
        args.flex = True
    if args.opt_epochs > 0:
        args.epochs = args.opt_epochs
    if args.load_stlp == True:
        args.load_tj = True
    if args.rect_head:
        args.interval = True
        args.diffusion_clip = True
        args.diff_full = True
    args.offline = not args.collect_data
    if args.test:
        args.epochs = 1
    
    return args


if __name__ == "__main__":
    args = generate_parser()    
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.3f seconds"%(t2-t1)) 