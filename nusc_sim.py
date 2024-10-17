import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from stl_d_lib import *
import utils
from utils import uniform, to_np, dict_to_cuda

from queue import Queue
import threading
import pickle

import nusc_api as napi

from nusc_train import generate_parser, get_dataloader, build_stl_cache, get_neighbor_trajs,\
     infer_gt_stlp, generate_trajs, dynamics, diffusion_rollout,\
     compute_stl_dense, augment_batch_data, compute_shortest_dist_refined, \
     pre_prepare_stl_cache, get_diffusion_coeffs, mask_mean

from nusc_viz import plot_agent, get_nusc_color_map
from nusc_model import Net

class NuScenesSim:
    def __init__(self, nusc, nusc_map_d, meta_d, args):
        self.args = args
        self.nusc = nusc
        self.nusc_map_d = nusc_map_d
        self.meta_d = meta_d
        self.epi = 0
        self.ti = 0
    
    def pre_check(self, batch):
        gt_trajs = batch["ego_traj"][..., :6]
        if torch.mean(gt_trajs[:, 3])<1.0:
            print("Average speed too slow, skip...")
            return False
        return True

    def reset(self, batch=None):
        args = self.args
        nusc = self.nusc
        nusc_map_d = self.nusc_map_d
        if batch is not None:
            i = 0
            self.traj_i = batch["traj_i"][i].item()
            self.base_ti = batch["ti"][i].item()
            
            self.gt_trajs = batch["ego_traj"][..., :6]  # contain the heading angle
            self.neighbor_trajs = batch["neighbors_traj"]
            if args.gt_nei:
                self.neighbor_trajs_est = batch["neighbors_traj"]
            else:
                self.neighbor_trajs_est = batch["neighbor_trajs_aug"]

            self.gt_stlp = infer_gt_stlp(batch, self.gt_trajs, args)
            self.traj_total_len = len(self.meta_d[self.traj_i])
            self.gt_trajs_np = to_np(self.gt_trajs)

            self.sim_state = self.gt_trajs[:, 0, :4]

            # NuScenes api
            if nusc is not None:
                self.my_scene = nusc.scene[self.traj_i]
                self.nusc_map = nusc_map_d[nusc.get("log", self.my_scene["log_token"])["location"]]
            else:
                self.my_scene = None
                self.nusc_map = None

            # statistics
            self.bad_cnt = 0
            self.collide = False
            self.out_of_lane = False
            self.backup_cnt = 0
            self.sim_length = 0
        else:
            raise NotImplementedError

        # keep batch shape
        sample_d = {
            "ego_state": self.sim_state,
            
            "neighbors": self.neighbor_trajs[:, :, 0],
            "neighbor_trajs": self.neighbor_trajs,
            "neighbor_trajs_aug": self.neighbor_trajs_est,

            "currlane_wpts": batch["currlane_wpts"],
            "leftlane_wpts": batch["leftlane_wpts"],
            "rightlane_wpts": batch["rightlane_wpts"],
            "curr_id": batch["curr_id"],
            "left_id": batch["left_id"],
            "right_id": batch["right_id"],

            "gt_stlp": self.gt_stlp,
            "ego_traj": self.gt_trajs,

            "gt_high_level": batch["gt_high_level"],
        }
        self.bad_cnt = 0
        self.epi += 1
        self.ti = 0
        self.trajs = [self.sim_state * 1.0]

        self.figname_list = []
        return sample_d

    def step(self, u, ego_traj):
        args = self.args
        nusc = self.nusc
        nusc_map_d = self.nusc_map_d
        nusc_map = self.nusc_map
        D_SAFE = 0.1

        # collect the action
        
        # update the ego
        new_sim_state = self.sim_state + dynamics(self.sim_state, u) * args.dt
        self.trajs.append(new_sim_state * 1.0)

        # find the neighbor
        sample_d = {"ego_state": new_sim_state[0]}
        sample_token = self.meta_d[self.traj_i][self.ti+1]
        sample_d["neighbors"], nearest_ann_tokens = napi.get_nearest_neighbors(
            nusc, sample_token, new_sim_state[0].cpu(), args.n_neighbors, ret_full=True)

        if args.gt_nei:
            tokens_nt = self.meta_d[self.traj_i][self.ti+1:self.ti+1+args.nt]
            for iii in range(len(tokens_nt), args.nt):
                tokens_nt.append("PLACEHOLDER_%02d"%(iii))
            sample_d["neighbor_trajs_aug"], sample_d["neighbors_traj_idx"] = napi.get_neighbor_trajectories(
                nusc, sample_token, tokens_nt, new_sim_state[0].cpu(), 
                k=self.args.n_neighbors, dt=args.dt, nearest_ann_tokens=nearest_ann_tokens)
        else:
            sample_d["neighbor_trajs_aug"] = get_neighbor_trajs(sample_d["neighbors"][None, :], args.nt, args.dt, full=True)[0]

        # find the roads
        token_name = self.my_scene["first_sample_token"]
        dataroot=utils.get_data_dir()
        with open("%s/%s.pickle"%(os.path.join(dataroot, args.anno_path), token_name), "rb") as ff:
            anno_data = pickle.load(ff)
        
        sample_d["ego_traj"] = to_np(ego_traj)

        curr_id, currlane_wpts, currlane_full, left_id, leftlane_wpts, leftlane_full, \
            right_id, rightlane_wpts, rightlane_full = napi.get_centerlines(nusc, nusc_map, \
                sample_token, self.ti+1, sample_d["ego_traj"], anno_data, args.n_expands, args.n_segs, ret_full=True)

        sample_d["currlane_wpts"] = currlane_wpts
        sample_d["leftlane_wpts"] = leftlane_wpts
        sample_d["rightlane_wpts"] = rightlane_wpts
        sample_d["curr_id"] = torch.tensor([(curr_id!=-1) * 1.0])
        sample_d["left_id"] = torch.tensor([(left_id!=-1) * 1.0])
        sample_d["right_id"] = torch.tensor([(right_id!=-1) * 1.0])
        sample_d["gt_traj"] = napi.get_ego_trajectory(nusc, self.meta_d[self.traj_i][self.ti+1: self.ti+1+args.nt], args.dt, return_numpy=True)
        sample_d["gt_high_level"] = napi.get_high_level_behaviors(nusc, anno_data, self.ti+1, args.nt, sample_d, sample_d["gt_traj"])

        # TODO compute for new gt_stlp
        if sample_d["gt_traj"].shape[0] < args.nt:
            real_nt = sample_d["gt_traj"].shape[0]
            new_traj = np.zeros((args.nt, 6))
            new_traj[:real_nt] = sample_d["gt_traj"][:real_nt]

            # constant heading and velocity
            new_traj[real_nt:args.nt, 2:6] = new_traj[real_nt-1:real_nt, 2:6]
            for ttti in range(real_nt, args.nt):
                new_traj[ttti, 0] = new_traj[ttti-1, 0] + new_traj[ttti-1, 3] * np.cos(new_traj[ttti-1, 2]) * args.dt
                new_traj[ttti, 1] = new_traj[ttti-1, 1] + new_traj[ttti-1, 3] * np.sin(new_traj[ttti-1, 2]) * args.dt
                
        self.gt_trajs_np = sample_d["gt_traj"]
        self.gt_trajs = torch.from_numpy(self.gt_trajs_np).float().cuda()
        
        # prepare for new observation
        sample_d_unsqueeze = {}
        for k in sample_d:
            if hasattr(sample_d[k],"device"):
                sample_d_unsqueeze[k] = sample_d[k].unsqueeze(0).float()
                
            elif isinstance(sample_d[k], np.ndarray):
                sample_d_unsqueeze[k] = torch.from_numpy(sample_d[k][None,:]).float()
            else:
                sample_d_unsqueeze[k] = torch.tensor([sample_d[k]])

        sample_d_unsqueeze = dict_to_cuda(sample_d_unsqueeze)

        I_VAL = 0
        I_X = 0

        # update statistics
        collide = False
        out_of_lane = False
        for nei in sample_d["neighbors"]:
            if nei[0]>0.5:
                closest_dist = compute_shortest_dist_refined(
                    new_sim_state[..., I_X:I_X+6].cpu(), nei[None, I_X+1:I_X+6+1], nei[None, I_VAL], 
                    ego_L=args.ego_L, ego_W=args.ego_W, nL=args.refined_nL, nW=args.refined_nW
                )
                if closest_dist < D_SAFE:
                    print("Collide!")
                    collide=True
                    if self.bad_cnt==0:
                        self.bad_cnt+=1
                        break
        
        # if driving out of road
        query_layers = nusc_map.layers_on_point(sample_d["ego_state"][0], sample_d["ego_state"][1])
        if query_layers["drivable_area"]=="":
            print("Drive out of lane!")
            out_of_lane=True
            if self.bad_cnt==0:
                self.bad_cnt+=1

        self.ti += 1
        self.sim_state = new_sim_state

        # viz should be here

        info = {
            "collide": collide,
            "out_of_lane": out_of_lane,
        }

        done = info["collide"] or info["out_of_lane"] or self.ti>=self.traj_total_len-2

        observation = sample_d_unsqueeze

        return observation, None, done, info

    def render(self, sample_d, plan_traj=None, diffusion_trajs=None, scores_all=None):
        PAPER=True
        ALPHA=1.0
        LW = 3.5
        LW_NEI = 3.5
        COLOR_AGENT = "#004E9E"
        COLOR_NEI = "#C04F15"
        COLOR_END = "#fb9a99"

        args = self.args
        nusc = self.nusc
        nusc_map_d = self.nusc_map_d
        nusc_map = self.nusc_map
        nusc_map.explorer.color_map["lane"] = "#FFFFFF"
        color_list=["blue", "green", "red"]

        if PAPER:
            nusc_map.explorer.color_map = get_nusc_color_map()
            del_list = ['road_divider', 'lane_divider', 'traffic_light']
        else:
            del_list = []

        i=0

        sim_state_np = to_np(self.sim_state[0])

        ego_xy, ego_th, ego_v, ego_L, ego_W = sim_state_np[:2], sim_state_np[2], sim_state_np[3], args.ego_L, args.ego_W

        r = 40
        my_patch = (ego_xy[0]-r,  ego_xy[1]-r, ego_xy[0]+r, ego_xy[1]+r)

        

        fig, ax = nusc_map.render_map_patch(my_patch, 
                                [xx for xx in nusc_map.non_geometric_layers if xx not in ['traffic_light', 'walkway', "ped_crossing", "stop_line"]+del_list], 
                                alpha=0.3, figsize=(8, 8) if args.diffusion else (8, 8), bitmap=None,
                                render_egoposes_range=(PAPER==False), render_legend=(PAPER==False)
                                )                
        bev_handles, bev_labels = ax.get_legend_handles_labels()

        cur_token = self.meta_d[self.traj_i][self.ti]
        neighbors, ann_tokens = napi.get_neighbors(nusc, cur_token, ret_full=True)
        if PAPER==False:   
            for nei in neighbors:
                plot_agent((nei[0], nei[1]), nei[2], nei[4] * 0.9, nei[5] * 0.9, ax, color="gray", alpha=0.5, arrow=(PAPER==False), edgecolor="black")

        # plot focus neighbors
        neighbors = to_np(sample_d["neighbors"][i])
        for ii in range(neighbors.shape[0]):
            if neighbors[ii, 0] == 1:
                nei = neighbors[ii, 1:]
                plot_agent((nei[0], nei[1]), nei[2], nei[4] * 1., nei[5] * 1., ax, color=COLOR_NEI, alpha=0.3, arrow=(PAPER==False), edgecolor="black")
        
        # plot current centerlines
        currlane = to_np(sample_d["currlane_wpts"][i].reshape((args.n_segs, 3)))
        leftlane = to_np(sample_d["leftlane_wpts"][i].reshape((args.n_segs, 3)))
        rightlane = to_np(sample_d["rightlane_wpts"][i].reshape((args.n_segs, 3)))     
        if PAPER==False:   
            plt.plot(currlane[:, 0], currlane[:, 1], "blue", linewidth=6, alpha=0.4, label="currlane")
            plt.plot(leftlane[:, 0], leftlane[:, 1], "green", linewidth=6, alpha=0.4, label="leftlane")
            plt.plot(rightlane[:, 0], rightlane[:, 1], "red", linewidth=6, alpha=0.4, label="rightlane")

        # plot the current agent
        plot_agent(ego_xy, ego_th, ego_L, ego_W, ax, color=COLOR_AGENT, arrow=(PAPER==False), edgecolor="black")

        # plot the currently taken traj        
        sim_trajs_np = to_np(torch.stack(self.trajs, dim=1))[0, :, :]
        plt.plot(sim_trajs_np[:, 0], sim_trajs_np[:, 1], color=COLOR_END if PAPER else "gray", alpha=1.0, linewidth=LW, zorder=1000, label="sim-traj")

        # plot the planned trajectory  
        if plan_traj is not None:
            plan_traj_np = to_np(plan_traj)
            plt.plot(plan_traj_np[0, :, 0], plan_traj_np[0, :, 1], color="purple", alpha=0.95, linewidth=LW+0.5, zorder=1500, label="plan-traj")

        if PAPER==False:   
            plt.plot(self.gt_trajs_np[:, 0], self.gt_trajs_np[:, 1], color="cyan", alpha=1, linewidth=2, zorder=1200, label="gt-traj")

        # plot the diffusion trajectory
        if diffusion_trajs is not None:
            nn_trajs = diffusion_trajs
            nn_trajs_np = to_np(nn_trajs.reshape((-1, args.n_randoms, 3)+nn_trajs.shape[-2:])[i])

            scores_argmax_i = torch.max(scores_all.reshape(-1,), dim=0)[1]
            max_score = torch.max(scores_all)

            for ii in range(args.n_randoms):
                for kk in range(3):
                    if (kk==0 and sample_d["curr_id"].item()==1) or (kk==1 and sample_d["left_id"].item()==1) or (kk==2 and sample_d["right_id"].item()==1):
                        plt.plot(nn_trajs_np[ii, kk, :, 0], nn_trajs_np[ii, kk, :, 1], 
                             color=color_list[kk], alpha=1 if PAPER else 0.8, linewidth=LW if PAPER else 1, 
                             zorder=800, label="diffusion (mode=%d)"%(kk) if ii==0 else None)
                    
                    if ii * 3 + kk == scores_argmax_i:
                        if PAPER==False:
                            plt.text(nn_trajs_np[ii, kk, -1, 0], nn_trajs_np[ii, kk, -1, 1], "max_s:%.2f"%max_score)
        
        if PAPER:
            plt.tick_params(left=False, right=False, labelleft=False, labelbottom = False, bottom = False) 
            plt.grid(False)
        else:
            ax.legend(frameon=True, loc='upper right')

        plt.axis("scaled")
        x_min, y_min, x_max, y_max = my_patch
        if PAPER:
            x_margin = np.minimum(x_max - x_min / 6, 5)
            y_margin = np.minimum(y_max - y_min / 6, 5)
            x_margin = y_margin = min(x_margin, y_margin)
        else:
            x_margin = np.minimum(x_max - x_min / 4, 50)
            y_margin = np.minimum(y_max - y_min / 4, 10)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        figname = "%s/viz_cl_epi%04d_tr%03d_i%03d_t%03d.png"%(args.viz_dir, self.epi, self.traj_i, self.base_ti, self.ti)

        utils.plt_save_close(figname)
        
        self.figname_list.append(figname)

    def render_trajs(self, args):
        return

    def render_gif(self):
        utils.generate_gif("%s/viz_cl_epi%04d_tr%03d_i%03d.gif"%(args.viz_dir, self.epi, self.traj_i, self.base_ti), duration=100, fs_list=self.figname_list)


def main():
    global args
    args = utils.setup_exp_and_logger(args, test=args.test)
    train_loader, val_loader, nusc, nusc_map_d, meta_list = get_dataloader(args)
    stls_cac = build_stl_cache(args)

    if all([not args.skip_nusc_load, nusc is None]):
        result_queue = Queue()
        thread_nusc = threading.Thread(target=napi.get_nuscenes, args=(args.mini, result_queue))
        thread_nusc.start()
    
    net = Net(args).cuda()
    if args.net_pretrained_path is not None:
        net.load_state_dict(torch.load(utils.get_model_path(args.net_pretrained_path)))

    if nusc is None and not args.skip_nusc_load:
        print("Wait for nuscene loading complete...")
        thread_nusc.join()
        nusc, nusc_map_d = result_queue.get()
    else:
        nusc, nusc_map_d = None, None
    
    nusc_sim = NuScenesSim(nusc, nusc_map_d, train_loader.dataset.meta_d, args)

    coeffs = get_diffusion_coeffs(args)

    loaders = {"train": train_loader, "val": val_loader}
    # for mode in ["val"]:
    mode = "val"
    data_loader = loaders[mode]
    # meters
    metrics = {"out_of_lane":[], "collide":[], "traj_len":[], "progress":[], "stl_acc":[], "area":[], "t":[]}
    for bi, batch in enumerate(data_loader):
        # start the close loop testing from here
        if nusc_sim.pre_check(batch)==False:
            continue
        bs = 1
        done = False
        bad_cnt = 0
        batch_cuda = dict_to_cuda(batch)
        if args.gt_nei:
            batch_cuda["neighbor_trajs_aug"] = batch_cuda["neighbors_traj"]
        else:
            batch_cuda["neighbor_trajs_aug"] = get_neighbor_trajs(batch_cuda["neighbors"], args.nt, args.dt, full=True)

        observation = nusc_sim.reset(batch_cuda)

        gt_stlp = observation["gt_stlp"]
        stlp_mul = gt_stlp.unsqueeze(1).repeat(1, args.n_randoms*3, 1).reshape(bs*args.n_randoms*3, 6)

        for key in metrics:
            metrics[key].append([])
        traj_len = 0
        while done == False:
            ########################
            ###  get the action  ###
            ########################
            # sample lots of gt_stlp
            # send to network
            # get action, generate trajectory
            # send to refine net
            # get the action
            # send to online refinement
            # send to backup control policy
            # derive the final policy

            ti = nusc_sim.ti            
            states = observation["ego_state"][:, :4]
            dense_states = states.unsqueeze(1).unsqueeze(1).repeat(1, args.n_randoms, 3, 1)
            n = bs * args.n_randoms * 3
            dense_states_flat = dense_states.reshape(n, dense_states.shape[-1])
            highlevel_dense = torch.tensor([0, 1.0, 2.0]).reshape(1, 3, 1).repeat(n//3, 1, 1).reshape(n, 1).cuda().float()

            new_batch = {
                "ego_traj": observation["ego_traj"],
                "neighbors": observation["neighbors"],
                "currlane_wpts": observation["currlane_wpts"],
                "leftlane_wpts": observation["leftlane_wpts"],
                "rightlane_wpts": observation["rightlane_wpts"],
                "curr_id": observation["curr_id"],
                "left_id": observation["left_id"],
                "right_id": observation["right_id"],
                "neighbor_trajs_aug": observation["neighbor_trajs_aug"],
                "gt_high_level": observation["gt_high_level"],
            }

            new_batch = augment_batch_data(new_batch, gt_stlp, args, args.n_randoms, stlp_dense=None)
            
            if args.test_aggressive:
                if bi==0:
                    new_batch["stlp_dense"][..., 0:1] = 0.0
                    new_batch["stlp_dense"][..., 1:2] = 1.0
                    new_batch["stlp_dense"][..., 2:3] = -1.0
                    new_batch["stlp_dense"][..., 3:4] = 2.0
                    new_batch["stlp_dense"][..., 4:5] = 2
                    new_batch["stlp_dense"][..., 5:6] = 0.2
                elif bi==1:
                    new_batch["stlp_dense"][..., 0:1] = 0.0
                    new_batch["stlp_dense"][..., 1:2] = 4.0
                    new_batch["stlp_dense"][..., 2:3] = -1.0
                    new_batch["stlp_dense"][..., 3:4] = 1.0
                    new_batch["stlp_dense"][..., 4:5] = 1
                    new_batch["stlp_dense"][..., 5:6] = 0.2
                elif bi==2:
                    new_batch["stlp_dense"][..., 0:1] = 0.0
                    new_batch["stlp_dense"][..., 1:2] = 6.0
                    new_batch["stlp_dense"][..., 2:3] = -1.0
                    new_batch["stlp_dense"][..., 3:4] = 1.0
                    new_batch["stlp_dense"][..., 4:5] = 0.2
                    new_batch["stlp_dense"][..., 5:6] = 0.2
            else:
                new_batch["stlp_dense"][..., 0:1] = 1.0
                new_batch["stlp_dense"][..., 1:2] = 9.0
                new_batch["stlp_dense"][..., 2:3] = -3.0 #-3.0
                new_batch["stlp_dense"][..., 3:4] = 2.0  #2.0
                new_batch["stlp_dense"][..., 4:5] = 0.1
                new_batch["stlp_dense"][..., 5:6] = 0.2

            tttt1=time.time()
            if args.diffusion:
                noise = torch.normal(0, 1, (n, args.nt * 2)).cuda().float()
                feature = None
                
                guidance_extras = (new_batch, dense_states_flat.detach(), stls_cac) if args.guidance else None

                resres = diffusion_rollout(noise, net, new_batch, highlevel_dense, feature, args, coeffs, return_feature=True, guidance_extras=guidance_extras, maximize=True)
                if args.diff_full:
                    nn_controls, feature, nn_controls_list = resres
                else:
                    nn_controls, feature = resres
                    nn_controls_list = None
            elif args.vae:
                N = bs * args.n_randoms * 3
                rand_w0 = uniform(-args.mul_w_max, args.mul_w_max, (bs, args.n_randoms, 3, args.nt)) * 0.1
                rand_a0 = uniform(-args.mul_a_max, args.mul_a_max, (bs, args.n_randoms, 3, args.nt))
                new_hint = torch.stack([rand_w0, rand_a0], dim=-1).cuda() # TODO
                new_hint = new_hint.reshape(N, args.nt * 2)

                new_batch["params_init"] = new_hint
                ext = {"highlevel": highlevel_dense}
                gaussian_sample = torch.normal(0, 1, (bs * args.n_randoms * 3, args.vae_dim)).to(highlevel_dense.device).float()
                results = net(new_batch, ext=ext, n_randoms=args.n_randoms, sample=gaussian_sample)
                nn_controls = results[0]

            elif args.bc:
                N = bs * args.n_randoms * 3
                rand_w0 = uniform(-args.mul_w_max, args.mul_w_max, (bs, args.n_randoms, 3, args.nt)) * 0.1
                rand_a0 = uniform(-args.mul_a_max, args.mul_a_max, (bs, args.n_randoms, 3, args.nt))
                new_hint = torch.stack([rand_w0, rand_a0], dim=-1).cuda() # TODO
                new_hint = new_hint.reshape(N, args.nt * 2)
                new_batch["params_init"] = new_hint
                ext = {"highlevel": highlevel_dense}
                nn_controls = net(new_batch, ext=ext, n_randoms=args.n_randoms)
            else:
                raise NotImplementedError
            
            nn_trajs = generate_trajs(dense_states_flat, nn_controls, args.dt)  # (n, nt, 6?)
            
            if (args.diffusion and args.not_use_rect==False and args.multi_cands is not None and args.rect_head)==False:
                prev_stl_input = pre_prepare_stl_cache(new_batch, dense_trajs=nn_trajs[:, :-1])
                _, prev_scores, prev_acc = compute_stl_dense(prev_stl_input, stls_cac, new_batch["highlevel_dense"], prev_stl_input["dense_valids"], args)
            
            if args.diffusion and args.rect_head and args.not_use_rect==False:
                if args.multi_cands is not None:
                    # cat in the first dim (multi_cands, N, ...)
                    states_mul = dense_states_flat.repeat(args.multi_cands, 1)
                    nn_ctrls_mul = torch.cat(nn_controls_list[-args.multi_cands:], dim=0)
                    nn_trajs_mul = generate_trajs(states_mul, nn_ctrls_mul, args.dt)
                    prev_stl_input = pre_prepare_stl_cache(new_batch, dense_trajs=nn_trajs_mul[:, :-1], repeat_n=args.multi_cands)
                    _, scores_hist_list, prev_acc = compute_stl_dense(prev_stl_input, stls_cac, 
                                        new_batch["highlevel_dense"].repeat((args.multi_cands, *[1]*(new_batch["highlevel_dense"].dim()-1))), 
                                        prev_stl_input["dense_valids"].reshape(-1), args)

                    controls_hist_list = nn_ctrls_mul.reshape(args.multi_cands, nn_controls_list[-1].shape[0], args.nt, 2)
                    scores_hist_list = scores_hist_list.reshape(args.multi_cands, nn_controls_list[-1].shape[0])
                    scores_hist_max, scores_hist_max_i = torch.max(scores_hist_list, dim=0)
                    controls_hist_max = controls_hist_list[scores_hist_max_i, range(scores_hist_max_i.shape[0])]
                    nn_controls = controls_hist_max
                    prev_scores = scores_hist_max

                rect_controls = net.rect_forward(feature, highlevel_dense, new_batch["stlp_dense"][:,0], nn_controls.detach(), prev_scores.detach(), extras=nn_controls_list)            
                if args.n_rolls is not None:
                    for _ in range(args.n_rolls):
                        prev_trajs_re = generate_trajs(dense_states_flat, rect_controls, args.dt)
                        prev_stl_input_re = pre_prepare_stl_cache(new_batch, dense_trajs=prev_trajs_re[:,:-1])
                        _, prev_scores_re, prev_acc_re = compute_stl_dense(prev_stl_input_re, stls_cac, new_batch["highlevel_dense"], prev_stl_input_re["dense_valids"], args)
                        rect_controls = net.rect_forward(feature, highlevel_dense, new_batch["stlp_dense"][:,0], rect_controls.detach(), prev_scores_re.detach(), extras=nn_controls_list)

                rect_trajs = generate_trajs(dense_states_flat, rect_controls, args.dt)  # (n, nt, 6?)

                stl_input = pre_prepare_stl_cache(new_batch, dense_trajs=rect_trajs[:, :-1])

                _, scores_all, acc = compute_stl_dense(stl_input, stls_cac, new_batch["highlevel_dense"], stl_input["dense_valids"], args)
                scores = scores_all
                nn_controls = rect_controls
                nn_trajs = rect_trajs

                max_score_find = torch.max(scores_all.reshape(n//3, 3)[:, 0:1])
                if args.lite_refine:
                    need_refinement = (max_score_find <= 0)
                else:
                    need_refinement = True
                if need_refinement:
                    N = bs * args.n_randoms * 3
                    if args.refinement:
                        K = 6
                        N_ITERS = 50
                        STL_THRES = 0.0005
                        LR = 3e-1 #1e-1
                        lamdas = torch.ones(N, K).cuda().requires_grad_()
                        optimizer = torch.optim.Adam([lamdas], lr=LR)
                        states_flat_new = dense_states_flat.detach()

                        nn_trajs = generate_trajs(states_flat_new, nn_controls, args.dt).reshape(N, args.nt+1, 4)
                        stl_input = {
                            "ego_traj": nn_trajs[:, :-1],
                            "neighbors": new_batch["neighbors_dense"],
                            "currlane_wpts": new_batch["currlane_wpts_dense"],
                            "leftlane_wpts": new_batch["leftlane_wpts_dense"],
                            "rightlane_wpts": new_batch["rightlane_wpts_dense"],
                            "stlp": new_batch["stlp_dense"],
                            "dense_valids": new_batch["valids_dense"].reshape(-1, ),
                            "gt_high_level": new_batch["gt_high_level"],
                        }

                        scores_list, scores, acc = compute_stl_dense(stl_input, stls_cac, new_batch["highlevel_dense"], stl_input["dense_valids"], args)

                        violated = (torch.logical_and(scores<=0, new_batch["valids_dense"].reshape(-1, )>0)).float()
                        violated = violated.reshape(bs * args.n_randoms * 3, 1, 1)

                        print("before, acc=%.3f"%(acc))
                        for opt_i in range(N_ITERS):
                            ratios = torch.softmax(lamdas, dim=-1)
                            if K==2:
                                optim_controls = nn_controls.detach() * ratios[..., 0:1, None] + nn_controls_list[0].detach() * ratios[..., 1:2, None]
                            elif K==3:
                                optim_controls = nn_controls.detach() * ratios[..., 0:1, None] + nn_controls_list[80].detach() * ratios[..., 1:2, None]\
                                        + nn_controls_list[95].detach() * ratios[..., 2:3, None]
                            elif K==6:
                                optim_controls = nn_controls.detach() * ratios[..., 0:1, None] + nn_controls_list[0].detach() * ratios[..., 1:2, None]\
                                        + nn_controls_list[50].detach() * ratios[..., 2:3, None]\
                                        + nn_controls_list[80].detach() * ratios[..., 3:4, None]\
                                        + nn_controls_list[90].detach() * ratios[..., 4:5, None]\
                                        + nn_controls_list[95].detach() * ratios[..., 5:6, None]
                            
                            optim_controls = nn_controls.detach() * (1-violated.detach()) + violated.detach() * optim_controls
                            optim_trajs = generate_trajs(states_flat_new.detach(), optim_controls, args.dt)
                            optim_stl_input = {
                                "ego_traj": optim_trajs[:, :-1],
                                "neighbors": new_batch["neighbors_dense"].detach(),
                                "currlane_wpts": new_batch["currlane_wpts_dense"].detach(),
                                "leftlane_wpts": new_batch["leftlane_wpts_dense"].detach(),
                                "rightlane_wpts": new_batch["rightlane_wpts_dense"].detach(),
                                "stlp": new_batch["stlp_dense"].detach(),
                                "dense_valids": new_batch["valids_dense"].reshape(-1, ).detach(),
                                "gt_high_level": new_batch["gt_high_level"].detach(),
                            }
                            _, optim_scores, optim_acc = compute_stl_dense(optim_stl_input, stls_cac, new_batch["highlevel_dense"].detach(), optim_stl_input["dense_valids"].detach(), args)

                            loss = mask_mean(torch.nn.ReLU()(STL_THRES-optim_scores), new_batch["valids_dense"].reshape(-1, ).detach())
                            # loss = torch.mean(optim_trajs)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            print("OPTIM [%2d/%2d] loss:%.4f acc:%.4f"%(opt_i, N_ITERS, loss.item(), optim_acc.item()))
                            print("OPTIM %2d loss:%.4f"%(opt_i, loss.item()))
                        rect_controls = optim_controls.detach()
                        rect_trajs = optim_trajs.detach()
                        scores_all = optim_scores
                    elif args.raw_refinement:
                        N_ITERS = 5
                        STL_THRES = 0.0005
                        LR = 3e-2 #1e-1
                        res_controls = torch.zeros(N, args.nt, 2).cuda().requires_grad_()
                        optimizer = torch.optim.Adam([res_controls], lr=LR)
                        states_flat_new = dense_states_flat.detach()

                        violated = (torch.logical_and(scores<=0, new_batch["valids_dense"].reshape(-1, )>0)).float()
                        violated = violated.reshape(N, 1, 1)

                        print("before, acc=%.3f"%(acc))
                        for opt_i in range(N_ITERS):                  
                            optim_controls = nn_controls.detach() + violated.detach() * res_controls
                            optim_trajs = generate_trajs(states_flat_new.detach(), optim_controls, args.dt)
                            
                            optim_stl_input = {
                                "ego_traj": optim_trajs[:, :-1],
                                "neighbors": new_batch["neighbors_dense"].detach(),
                                "currlane_wpts": new_batch["currlane_wpts_dense"].detach(),
                                "leftlane_wpts": new_batch["leftlane_wpts_dense"].detach(),
                                "rightlane_wpts": new_batch["rightlane_wpts_dense"].detach(),
                                "stlp": new_batch["stlp_dense"].detach(),
                                "dense_valids": new_batch["valids_dense"].reshape(-1, ).detach(),
                                "gt_high_level": new_batch["gt_high_level"].detach(),
                            }

                            _, optim_scores, optim_acc = compute_stl_dense(optim_stl_input, stls_cac, new_batch["highlevel_dense"].detach(), optim_stl_input["dense_valids"].detach(), args)

                            stl_loss = mask_mean(torch.nn.ReLU()(STL_THRES-optim_scores), new_batch["valids_dense"].reshape(-1, ).detach())
                            reg_loss = stl_loss * 0
                            l2_loss = stl_loss * 0
                            loss = stl_loss + reg_loss + l2_loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                        rect_controls = optim_controls.detach()
                        rect_trajs = optim_trajs.detach()
                        scores_all = optim_scores
                else:
                    do_nothing=True
                ego_controls = rect_controls
                ego_trajs = rect_trajs  
            else:
                ego_controls = nn_controls
                ego_trajs = nn_trajs
                scores_all, acc = prev_scores, prev_acc
                stl_input = {k:v for k,v in prev_stl_input.items()}
            
            scores_all = scores_all.reshape(n//3, 3)
            scores_all[:, 1:3] = -10000
            total_idx = torch.argmax(scores_all)
            highest_score = scores_all.flatten()[total_idx]
            tttt2=time.time()
            sim_ctrl = ego_controls[total_idx].unsqueeze(0)
            sim_traj = ego_trajs[total_idx].unsqueeze(0)  # (1, 21, 4)
            sim_traj = torch.cat([sim_traj, args.ego_L * torch.ones_like(sim_traj[..., 0:1]), args.ego_W * torch.ones_like(sim_traj[..., 0:1])], dim=-1)

            if args.backup:
                nei_est = observation["neighbor_trajs_aug"]
                D_SAFE=0.1
                est_collide=False
                for ni in range(nei_est.shape[1]):
                    if nei_est[0,ni,0,0]>0.5:
                        closest_dist = compute_shortest_dist_refined(
                            sim_traj[0:1, 2, 0:0+6].cpu(), nei_est[0, ni:ni+1, 2, 0+1:0+7].cpu(), nei_est[0, ni:ni+1, 2, 0].cpu(),
                            ego_L=args.ego_L, ego_W=args.ego_W, nL=args.refined_nL, nW=args.refined_nW
                        )
                        if closest_dist < D_SAFE:
                            print("detect as unsafe: epi", bi, traj_len)
                            est_collide=True
                            u_res = solve_bak(
                                sim_traj[..., 0:3, 0:0+6], sim_ctrl[..., 0:3, 0:2], 
                                nei_est[0, ni:ni+1,0:3].cpu(),
                                ego_L=args.ego_L, ego_W=args.ego_W, nL=args.refined_nL, nW=args.refined_nW, dt=args.dt)
                            u_res = u_res.cuda()
                            sim_ctrl[:, :2] = sim_ctrl[:, :2] + u_res[None, :]
                            
                            sim_traj = generate_trajs(sim_traj[:, 0, 0:4], sim_ctrl, dt=args.dt)
                            sim_traj = torch.cat([sim_traj, args.ego_L * torch.ones_like(sim_traj[..., 0:1]), args.ego_W * torch.ones_like(sim_traj[..., 0:1])], dim=-1)
                            break

            observation, _, done, info = nusc_sim.step(sim_ctrl[0:1, 0, :2], ego_traj=sim_traj[0, :-1])
            
            traj_len += 1
            
            ma_std, ma_vol, ma_std_list, ma_vol_list = napi.measure_diversity(
                                    ego_trajs[..., :-1, :2].reshape(bs, args.n_randoms, 3, args.nt*2), 
                                    scores_all.reshape(bs, args.n_randoms, 3), 
                                    new_batch["valids_dense"].reshape(bs, args.n_randoms, 3), args.nt)

            results = napi.measure_extra_diversity(
                ego_trajs[:, :-1].reshape(bs, args.n_randoms, 3, args.nt*4), 
                scores_all.reshape(bs, args.n_randoms, 3), 
                new_batch["valids_dense"].reshape(bs, args.n_randoms, 3), args.nt,
                nn_controls.reshape(bs, args.n_randoms, 3, args.nt*2),
                -args.mul_w_max, args.mul_w_max, -args.mul_a_max, args.mul_a_max,
            )
            area = results["area"].item()

            # metrics
            metrics["collide"][-1].append(info["collide"] * 1.0)
            metrics["out_of_lane"][-1].append(info["out_of_lane"] * 1.0)
            metrics["traj_len"][-1].append(traj_len)
            metrics["progress"][-1].append(np.sum(to_np(torch.stack(nusc_sim.trajs, dim=1))[0, :, 3])*args.dt)
            metrics["stl_acc"][-1].append(torch.mean((scores_all[:, 0:1]>0).float()).item())
            metrics["area"][-1].append(area)
            metrics["t"][-1].append(tttt2-tttt1)

            if done:
                collide_rate = np.mean([xx[-1] for xx in metrics["collide"]])
                out_of_lane_rate = np.mean([xx[-1] for xx in metrics["out_of_lane"]])
                avg_traj_len = np.mean([xx[-1] for xx in metrics["traj_len"]])
                avg_progress = np.mean([xx[-1] for xx in metrics["progress"]])
                avg_stl_acc = np.mean(cat_list(metrics["stl_acc"]))
                avg_area = np.mean(cat_list(metrics["area"]))
                avg_t = np.mean(cat_list(metrics["t"]))
                print("### Traj:%04d ### len:%02d || compliance:%.3f area:%.3f progress:%.3f | coll:%.3f ool:%.3f avg_len:%.3f  | time:%.3f"%(
                    bi, traj_len, avg_stl_acc, avg_area, avg_progress,
                    collide_rate, out_of_lane_rate, avg_traj_len, avg_t
                ))

            if args.no_viz==False:
                if args.viz_last==False or done:
                    nusc_sim.render(observation, sim_traj, ego_trajs, scores_all=scores_all)

                if done:
                    nusc_sim.render_gif()

def solve_bak(ego_traj, ego_ctrls, nei_traj, ego_L, ego_W, nL, nW, dt, D_SAFE=0.1):
    # ego_traj  (1, 3, 6)
    # ego_ctrls (1, 3, 2)
    # nei_traj  (1, 3, 6)
    u_res = torch.zeros(2, 2).cuda().requires_grad_()
    niters=500
    lr=1e-2
    optimizer = torch.optim.Adam([u_res], lr=lr)
    for i in range(niters):
        # print(ego_traj.shape, ego_ctrls.shape, u_res.shape)
        new_traj = generate_trajs(ego_traj[:, 0, 0:4].detach(), ego_ctrls[:, 0:2].detach() + u_res[None,:], args.dt)
        closest_dist0 = compute_shortest_dist_refined(
                            new_traj[0, 1:3, 0:0+4].cpu(), nei_traj[0, 1:3, 0+1:0+7].detach(), ind=1.0,
                            ego_L=ego_L, ego_W=ego_W, nL=nL, nW=nW
                        ) # (1, 2)
        loss_d = torch.mean(torch.nn.ReLU()(D_SAFE*1.01-closest_dist0))
        loss_reg = torch.mean(torch.square(u_res))
        loss = loss_d + loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print("%03d loss:%.4f %.4f"%(i, loss.item(), loss_d.item()))
    
    return u_res


def cat_list(list_of_lists):
    final_list=[]
    for li in list_of_lists:
        final_list = final_list + li
    return final_list


if __name__ == "__main__":
    args = generate_parser()
    # args.anno_path = "../"+args.anno_path
    if args.cache_path is not None:
        args.offline = True
    
    args.batch_size = 1
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.3f seconds"%(t2-t1))