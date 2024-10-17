import os
import time
import numpy as np
import torch
import pickle
from stl_d_lib import *
import utils
from utils import uniform, dict_to_torch
import nusc_api as napi


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, nusc, nusc_map_d, meta_list, cache, split, args, ridx=None):
        self.nusc = nusc
        self.nusc_map_d = nusc_map_d
        self.meta_list = meta_list
        self.cache = cache
        self.split = split
        self.args = args
        
        self.meta_d = {traj_i:tokens for traj_i, tokens in self.meta_list}
        if args.generate_split_on_the_fly:
            self.indices = self.gen_indices_on_the_fly(split, ridx)
        else:
            self.indices = self.load_split_from_file(split, ridx)

        print("%s %d n_samples"%(self.split, len(self.indices)))

        self.pickle_cache = {}
    
    def __len__(self):
        return len(self.indices)

    def load_split_from_file(self, split, ridx):
        assert ridx is not None
        if self.args.filter_traj is not None:
            indices = [[3, 7,]]
            if self.args.test_scenes:
                indices = [
                    [0, 13],   # straight line some cars parking on the side
                    [4, 1],    # 
                    [5, 1],
                    [6, 1],
                    [27, 1],
                    [49, 1],
                    [56, 1],   # intersection, big truck
                    [58, 1],   # collide, at last
                    [74, 1],   # pure straight, 
                    [80, 1],   # straight case, high speed
                    [106, 1],  # dense case
                    [127, 1],  # interesting behavior (0) yield to let other car go
                    [128, 1],  # interesting behavior (1) yield to let other car go
                    [143, 1],  # interesting behavior (2) rear car emerge, accelerate
                    [177, 1],  # simple case, rear car acc
                    [179, 1],  # straight, lots of car, on the side,  
                    [185, 1],  # front car, so it needs to keep certain dist
                    [198, 1],  # front car, collide happens
                    [218, 1],  # front car, keep on a left turn lane
                    [228, 1],  # three lanes in the first half, can do highlevel?
                    [229, 1],  # three lanes in many steps, can do highlevel?
                    [252, 1],  # interesting, bypass big car
                    [300, 1],  # complex lanes, full three lines
                    [509, 1],  # rear car acc, and collide
                    [521, 1],  # straight line, bypass another car
                    [781, 1],  # roundabout
                ]
            elif self.args.test_aggressive:
                indices = [
                    [781, 1],
                    [781, 1],
                    [781, 1],
                ]
            indices = [(ind[0], ind[1], self.meta_d[ind[0]][ind[1]]) for ind in indices]

        else:
            if self.args.collect_data:
                split_list = ["train", "val"]
            else:
                split_list = [split]
            indices = []
            for split_item in split_list:
                file_path = "data/%s%s%s_split.txt"%(
                    "mini_" if self.args.mini else "", "mixed_", split_item)
                with open(file_path, "r") as f:
                    for line in f.readlines():
                        traj_i, ti, tokens_i = line.strip().split(" ")
                        if self.args.test_t1:
                            if int(ti)!=1:
                                continue
                        indices.append([int(traj_i), int(ti), tokens_i])
        return indices

    def gen_indices_on_the_fly(self, split, ridx):
        args = self.args
        indices = []
        for traj_i, tokens in self.meta_list:
            for ti in range(1, len(tokens) - self.args.nt + 1):
                indices.append((traj_i, ti, tokens[ti]))
        
        torch.manual_seed(args.seed)
        rridx = torch.randperm(len(indices))
        new_train_len = int(len(indices) * self.args.train_ratio)
        if split=="train":
            indices = [indices[idxx] for idxx in rridx[:new_train_len]]
        else:
            indices = [indices[idxx] for idxx in rridx[new_train_len:]]
        return indices

    def __getitem__(self, idx):
        ttt1=time.time()
        traj_i, ti, my_token = self.indices[idx]
        args = self.args
        if self.args.offline:
            sample_d = dict_to_torch(self.cache[traj_i][ti], keep_keys=["traj_i", "ti", "len_full"])
        else:
            sample_d = {"traj_i": traj_i, "ti": ti, "len_full": len(self.meta_d[traj_i])}
            
            nusc = self.nusc
            nusc_map_d = self.nusc_map_d
            my_scene = nusc.scene[traj_i]
            nusc_map = nusc_map_d[nusc.get("log", my_scene["log_token"])["location"]]

            ttt2=time.time()

            tokens_nt = self.meta_d[traj_i][ti:ti+self.args.nt]
            sample_d["ego_traj"] = napi.get_ego_trajectory(nusc, tokens_nt, self.args.dt, return_numpy=True)

            ttt3=time.time()

            ego_state = torch.from_numpy(sample_d["ego_traj"][0]).float()
            sample_d["neighbors"], nearest_ann_tokens = napi.get_nearest_neighbors(nusc, my_token, ego_state, k=self.args.n_neighbors, ret_full=True)

            if args.gt_nei:
                sample_d["neighbors_traj"], sample_d["neighbors_traj_idx"] = napi.get_neighbor_trajectories(nusc, my_token, tokens_nt, ego_state, 
                                                    k=self.args.n_neighbors, dt=args.dt, nearest_ann_tokens=nearest_ann_tokens)

            ttt4=time.time()

            token_name = my_scene["first_sample_token"]
            if token_name in self.pickle_cache:
                anno_data = self.pickle_cache[token_name]
            else:
                dataroot=utils.get_data_dir()
                with open("%s/%s.pickle"%(os.path.join(dataroot,args.anno_path), token_name), "rb") as ff:
                    anno_data = pickle.load(ff)
                    self.pickle_cache[token_name] = anno_data
            
            sample_d["gt_high_level"] = napi.get_high_level_behaviors(nusc, anno_data, ti, args.nt, sample_d, sample_d["ego_traj"])

            ttt5=time.time()

            curr_id, currlane_wpts, currlane_full, left_id, leftlane_wpts, leftlane_full, \
                right_id, rightlane_wpts,rightlane_full = napi.get_centerlines(nusc, nusc_map, \
                    my_token, ti, sample_d["ego_traj"], anno_data, self.args.n_expands, self.args.n_segs, ret_full=True, highlevel=sample_d["gt_high_level"])
            
            ttt6=time.time()

            #################
            # consider uturn
            uturn_status = -1 # (0, 1 | 2, 3 | 4, 5) for l/r-turn | l/r uturn | valid l/r uturn
            # how to detect possible uturn (compare the starting point)
            # when the non-curr lane is in current lane's oppo direction (closest point heading diff > np.pi/2)
            # consider the left case
            if left_id!=-1:
                if np.cos(leftlane_wpts[0, -1] - currlane_wpts[0, -1]) < 0:  # potential to be a uturn
                    valid_uturn = napi.is_able_uturn(nusc_map, ego_state, currlane_wpts, leftlane_wpts)
                    if valid_uturn:
                        uturn_status = 4
                    else:
                        uturn_status = 2
                        left_id = -1
                        leftlane_wpts = leftlane_wpts * 0                            
                else:
                    uturn_status = 0

            # consider the right case
            if right_id!=-1:
                if np.cos(rightlane_wpts[0, -1] - currlane_wpts[0, -1]) < 0:  # potential to be a uturn
                    valid_uturn = napi.is_able_uturn(nusc_map, ego_state, currlane_wpts, rightlane_wpts)
                    if valid_uturn:
                        uturn_status = 5
                    else:
                        uturn_status = 3
                        right_id = -1
                        rightlane_wpts = rightlane_wpts * 0
                else:
                    uturn_status = 1
            sample_d["uturn_status"] = torch.tensor([uturn_status])
            sample_d["currlane_wpts"] = currlane_wpts
            sample_d["leftlane_wpts"] = leftlane_wpts
            sample_d["rightlane_wpts"] = rightlane_wpts
            sample_d["curr_id"] = torch.tensor([(curr_id!=-1) * 1.0])
            sample_d["left_id"] = torch.tensor([(left_id!=-1) * 1.0])
            sample_d["right_id"] = torch.tensor([(right_id!=-1) * 1.0])

            for key in sample_d:           
                if isinstance(sample_d[key], np.ndarray):
                    sample_d[key] = torch.from_numpy(sample_d[key]).float()
            
            ttt7=time.time()                

        if hasattr(args,"exp_dir_full"):# and args.test==False:
            params_path = os.path.join(self.args.exp_dir_full, "models", "params_%05d_%04d.npy"%(traj_i, ti))
            params_path2 = os.path.join(self.args.exp_dir_full, "models", "params_%05d_%04d_init.npy"%(traj_i, ti))
            if os.path.exists(params_path):
                sample_d["params"] = torch.from_numpy(np.load(params_path)).float()
                sample_d["params_init"] = torch.from_numpy(np.load(params_path2)).float()
            else:
                if args.params_load_path is not None:
                    params_path = os.path.join(self.args.exp_dir_full, "../../" if args.test else "../", args.params_load_path, "models", "params_%05d_%04d.npy"%(traj_i, ti))
                    params_path2 = os.path.join(self.args.exp_dir_full, "../../" if args.test else "../", args.params_load_path, "models", "params_%05d_%04d_init.npy"%(traj_i, ti))
                    sample_d["params"] = torch.from_numpy(np.load(params_path)).float()
                    sample_d["params_init"] = torch.from_numpy(np.load(params_path2)).float()
                else:
                    rand_w0 = uniform(-args.mul_w_max, args.mul_w_max, (args.n_randoms, 3, args.nt)) * 0.1
                    rand_a0 = uniform(-args.mul_a_max, args.mul_a_max, (args.n_randoms, 3, args.nt))
                    sample_d["params"] = torch.stack([rand_w0, rand_a0], dim=-1)  # (M, 3, nt, 2)
                    sample_d["params_init"] = sample_d["params"].detach() * 1.0     
            
            if args.load_stlp:
                params_path_stlp = os.path.join(self.args.exp_dir_full, "../../" if args.test else "../", args.params_load_path, "models", "params_%05d_%04d_stlp.npy"%(traj_i, ti))
                sample_d["pre_stlp"] = torch.from_numpy(np.load(params_path_stlp)).float()

                params_path_scores = os.path.join(self.args.exp_dir_full, "../../" if args.test else "../", args.params_load_path, "models", "scores_%05d_%04d.npy"%(traj_i, ti))
                sample_d["tj_scores_prior"] = torch.from_numpy(np.load(params_path_scores)).float()

        '''
        params_init torch.Size([64, 3, 20, 2])
        params torch.Size([64, 3, 20, 2])
        pre_stlp torch.Size([64, 3, 1, 6])
        tj_scores_prior torch.Size([64, 3])
        '''
        original_n_randoms = sample_d["params_init"].shape[0]
        if original_n_randoms != self.args.n_randoms:
            sample_idx = np.random.choice(list(range(original_n_randoms)), self.args.n_randoms)
            sample_d["params_init"] = sample_d["params_init"][sample_idx]
            sample_d["params"] = sample_d["params"][sample_idx]
            if args.load_stlp:
                sample_d["pre_stlp"] = sample_d["pre_stlp"][sample_idx]
                sample_d["tj_scores_prior"] = sample_d["tj_scores_prior"][sample_idx]

        ttt8=time.time()
        # print("%.3f | %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (ttt8 - ttt1, ttt2-ttt1, ttt3-ttt2, ttt4-ttt3, ttt5-ttt4, ttt6-ttt5, ttt7-ttt6, ttt8-ttt7))
        return sample_d