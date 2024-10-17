import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
from matplotlib.ticker import PercentFormatter

import itertools

from stl_d_lib import *
import utils
from utils import to_np, generate_bbox
from nusc_api import get_th_from_rotation

def plot_agent(xy, th, L, W, ax, color, alpha=1.0, arrow=True, edgecolor=None):
    bbox = generate_bbox(xy[0], xy[1], th, L, W)
    if edgecolor is not None:
        polygon = Polygon(bbox, facecolor=color, edgecolor=edgecolor, zorder=999)
    else:
        polygon = Polygon(bbox, color=color)
    ax.add_patch(polygon)
    if arrow:
        plt.plot([xy[0], xy[0] + L/2*np.cos(th)], [xy[1], xy[1] + L/2*np.sin(th)], color="yellow", zorder=1000, alpha=alpha)

def find_map(scene_id, nusc, nusc_map_d):
    my_scene = nusc.scene[scene_id]
    log = nusc.get("log", my_scene["log_token"])
    location = log["location"]
    nusc_map = nusc_map_d[location]
    return nusc_map

def compute_ctrls(trajs, dt):
    return (trajs[:, 1:, 2:4] - trajs[:, :-1, 2:4]) / dt

def hold_out(data, keep_ratio):
    assert len(data.shape)==1
    n = data.shape[0]
    remove_k = int((1-keep_ratio)/2 * n) # (remove top/low k entries) 
    ind1 = np.argpartition(data, remove_k)[:remove_k]
    ind2 = np.argpartition(data, -remove_k)[-remove_k:]
    ind_to_remove = np.union1d(ind1, ind2)
    ind_all = np.array(list(range(n)))
    ind_res = np.setdiff1d(ind_all, ind_to_remove)
    return ind_res, data[ind_res]

def plot_histograms(viz_dir, epi, viz_cache, dt):
    # 1. Plot the gt/nn histograms
    nbins=30
    label_list = ["Train-w", "Train-a", "Val-w", "Val-a"]
    ylabel_list = ["Groundtruth", "NN Estimation"]
    gt_ctrls_train = compute_ctrls(viz_cache["train"]["ego_traj"], dt).reshape(-1, 2)
    gt_ctrls_val = compute_ctrls(viz_cache["val"]["ego_traj"], dt).reshape(-1, 2)
    nn_ctrls_train = compute_ctrls(viz_cache["train"]["nn_trajs"], dt).reshape(-1, 2)
    nn_ctrls_val = compute_ctrls(viz_cache["val"]["nn_trajs"], dt).reshape(-1, 2)
    # gt_w_train, gt_a_train, gt_w_val, gt_a_val
    # nn_w_train, nn_a_train, nn_w_val, nn_a_val
    data_rig = list(itertools.chain(*[(xx[...,0], xx[...,1]) for xx in [gt_ctrls_train, gt_ctrls_val, nn_ctrls_train, nn_ctrls_val]]))
    for keep_ratio in [1, 0.9, 0.75]:
        f,axes = plt.subplots(2, 4, figsize=(8, 6))
        if keep_ratio!=1:
            data_rig_split = [hold_out(x, keep_ratio)[1] for x in data_rig]
        else:
            data_rig_split = [x * 1.0 for x in data_rig]
        the_ranges = [[min(np.min(data_rig_split[ii]),np.min(data_rig_split[ii+4])), max(np.max(data_rig_split[ii]),np.max(data_rig_split[ii+4]))] for ii in range(4)]
        for i in range(4): # control the columns
            axes[0, i].hist(data_rig_split[i], bins=nbins, color="blue", range=the_ranges[i], weights=np.ones(len(data_rig_split[i])) / len(data_rig_split[i]))
            axes[0, i].yaxis.set_major_formatter(PercentFormatter(1))
            axes[1, i].hist(data_rig_split[i+4], bins=nbins, color="red", range=the_ranges[i], weights=np.ones(len(data_rig_split[i+4])) / len(data_rig_split[i+4]))
            axes[1, i].yaxis.set_major_formatter(PercentFormatter(1))
            if i==0:
                axes[0, 0].set_ylabel(ylabel_list[0])
                axes[1, 0].set_ylabel(ylabel_list[1])
            axes[1, i].set_xlabel(label_list[i])                    
        plt.suptitle("Nuscenes controls output distribution (keep_ratio=%.2f)"%(keep_ratio))
        plt.tight_layout()
        utils.plt_save_close("%s/fig0_hist_e%03d_keep%.2f.png"%(viz_dir, epi, keep_ratio))

def plot_trajectories(viz_dir, epi, viz_cache, normalize_xyth_np):
    # gt_train, gt_val
    # nn_train, nn_val
    f,axes = plt.subplots(2,2, figsize=(8, 5))
    label_list = ["train", "val"]
    for i, mode in enumerate(viz_cache):
        for k in range(2):  # rows (gt/nn)
            ax = axes[k, i]
            trj = viz_cache[mode]["ego_traj"] if k==0 else viz_cache[mode]["nn_trajs"]
            trj = normalize_xyth_np(trj[:, :, :3], trj[:, 0:1, :3])
            for j in range(int(trj.shape[0]//10)):
                ax.plot(trj[j, :, 0], trj[j, :, 1], alpha=0.1, color="blue" if k==0 else "red", linewidth=2.0)
            ax.axis("scaled")
            # TODO this might need to change for a new dataset
            ax.set_xlim(-20, 200)
            ax.set_ylim(-80, 80)
            if k==1:
                ax.set_xlabel(label_list[i])
        axes[i, 0].set_ylabel("Trajectories")
    plt.suptitle("Nuscenes trajectory distribution")
    utils.plt_save_close("%s/fig1_traj_dist_e%03d.png"%(viz_dir, epi))

def get_nusc_color_map():
    return dict(drivable_area='#828282',
                             road_segment='#ffffff',
                             road_block='#627272',
                             lane='#ffffff',
                             ped_crossing='#fb9a99',
                             walkway='#e31a1c',
                             stop_line='#fdbf6f',
                             carpark_area='#aa4f00',
                             road_divider='#cab2d6',
                             lane_divider='#6a3d9a',
                             traffic_light='#7e772e')

def plot_paper_scene(nusc, nusc_map_d, meta_d, batch_np, dense_trajs_np, tj_scores_np, nn_trajs_np, scores_np, args, i, tj_n_randoms, nn_n_randoms, ego_only, opt_only):
    ALPHA=1.0
    LW = 3.5
    LW_NEI = 3.5
    COLOR_AGENT = "#004E9E"
    COLOR_NEI = "#C04F15"
    COLOR_END = "#fb9a99"

    traj_i = batch_np["traj_i"][i]
    ti = batch_np["ti"][i]
    nusc_map = find_map(batch_np["traj_i"][i], nusc, nusc_map_d)

    nusc_map.explorer.color_map = get_nusc_color_map()


    bs = batch_np["traj_i"].shape[0]

    ego_trajs = batch_np["ego_traj"][i]
    ego_xy, ego_th, ego_v, ego_L, ego_W = ego_trajs[0, :2], ego_trajs[0, 2], ego_trajs[0, 3], args.ego_L, args.ego_W #ego_trajs[0, 4], ego_trajs[0, 5]
    r = 50

    delta_r = 15
    
    # my_patch = (ego_xy[0]-r,  ego_xy[1]-r, ego_xy[0]+r, ego_xy[1]+r)
    my_patch = (ego_xy[0]+delta_r*np.cos(ego_th)-r,  ego_xy[1]+delta_r*np.sin(ego_th)-r, ego_xy[0]+delta_r*np.cos(ego_th)+r, ego_xy[1]+delta_r*np.sin(ego_th)+r)
    fig, ax = nusc_map.render_map_patch(my_patch, [xx for xx in nusc_map.non_geometric_layers if xx not in ['traffic_light', 'walkway', "ped_crossing", "stop_line"]+['road_divider', 'lane_divider', 'traffic_light'] ], 
                        alpha=0.3, figsize=(8, 8), bitmap=None, render_egoposes_range=False, render_legend=False)          
    
    # plot focus neighbors
    neighbors = batch_np["neighbors"][i]
    for ii in range(neighbors.shape[0]):
        if neighbors[ii, 0] == 1:
            nei = neighbors[ii, 1:]
            plot_agent((nei[0], nei[1]), nei[2], nei[4] * 1.0, nei[5] * 1.0, ax, color=COLOR_NEI, alpha=0.5, arrow=False, edgecolor="black")
            plt.plot(batch_np["neighbor_trajs_aug"][i, ii, :, 1], batch_np["neighbor_trajs_aug"][i, ii, :, 2], color=COLOR_NEI, alpha=ALPHA, linewidth=LW_NEI)

    # # plot current centerlines
    currlane = batch_np["currlane_wpts"][i].reshape((-1, 3))
    leftlane = batch_np["leftlane_wpts"][i].reshape((-1, 3))
    rightlane = batch_np["rightlane_wpts"][i].reshape((-1, 3))        

    lanes_d = {0:currlane, 1:leftlane, 2:rightlane}
    color_list=[COLOR_AGENT, "green", "red"]

    # dense_trajs_np = batch_np["dense_trajs"].reshape((-1, args.n_randoms, 3)+batch_np["dense_trajs"].shape[-2:])[i]
    dense_trajs_np = dense_trajs_np.reshape((-1, args.n_randoms, 3)+dense_trajs_np.shape[-2:])[i]
    plot_agent(ego_xy, ego_th, ego_L, ego_W, ax, color=COLOR_AGENT, arrow=False, edgecolor="black")
    high_level = batch_np["gt_high_level"][i, 0]

    # plt.plot(ego_trajs[:, 0], ego_trajs[:, 1], color="cyan", alpha=0.8, linewidth=2.5, zorder=500, label="gt_traj")

    if ego_only:
        nn_trajs_np = nn_trajs_np.reshape((-1, args.n_randoms, 3)+nn_trajs_np.shape[-2:])[i]
        for ii in range(args.n_randoms):
            for kk in range(3):
                if lanes_d[kk][0,0]!=0:
                    alpha=1.0
                    if args.viz_correct and scores_np.reshape(bs, args.n_randoms, 3)[i,ii,kk]<=0:
                        alpha=0.0
                        continue
                    plt.plot(nn_trajs_np[ii, kk, :, 0], nn_trajs_np[ii, kk, :, 1], color=color_list[kk], alpha=ALPHA*alpha, linewidth=LW, zorder=800, label="diffusion (mode=%d)"%(kk) if ii==0 else None)
                    # plt.scatter(nn_trajs_np[ii, kk, -1:, 0], nn_trajs_np[ii, kk, -1:, 1], color=COLOR_END, alpha=ALPHA*alpha, zorder=801)
    
    # plot multiple trajopt trajs
    if opt_only:
        for ii in range(args.n_randoms):
            for kk in range(3):
                if lanes_d[kk][0,0]!=0:
                    alpha=1.0
                    if args.viz_correct and tj_scores_np.reshape(bs, args.n_randoms, 3)[i,ii,kk]<=0:
                        alpha=0.0
                        continue
                    plt.plot(dense_trajs_np[ii, kk,:, 0], dense_trajs_np[ii, kk, :, 1], color=color_list[kk], alpha=ALPHA*alpha, linewidth=LW, zorder=800, label="trajopt (mode=%d)"%(kk) if ii==0 else None)
                    # plt.scatter(dense_trajs_np[ii, kk, -1:, 0], dense_trajs_np[ii, kk, -1:, 1], color=COLOR_END, alpha=ALPHA*alpha, zorder=801)
    # ax.legend(frameon=True, loc='upper right')
    # plt.tick_params(axis='both', which='both', left=False, bottom=False, top=False, labelbottom=False)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom = False, bottom = False) 
    plt.grid(False)
    plt.axis("scaled")
    x_min, y_min, x_max, y_max = my_patch
    x_margin = np.minimum(x_max - x_min / 6, 5)
    y_margin = np.minimum(y_max - y_min / 6, 5)
    x_margin = y_margin = min(x_margin, y_margin)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    if ego_only:
        plt.savefig("%s/viz_tr%03d_i%03d_diffusion.png"%(args.viz_dir, batch_np["traj_i"][i], batch_np["ti"][i]), bbox_inches='tight', pad_inches=0.03)
    elif opt_only:
        plt.savefig("%s/viz_tr%03d_i%03d_trajopt.png"%(args.viz_dir, batch_np["traj_i"][i], batch_np["ti"][i]), bbox_inches='tight', pad_inches=0.03)
    else:
        raise NotImplementedError
    plt.close()

def plot_nuscene_viz(i, split, viz_cache, epi, nusc, nusc_map_d, dataloader, multi_check=False, ego_only=False, opt_only=False, args=None):
    batch_np = viz_cache[split]
    traj_i = batch_np["traj_i"][i]
    ti = batch_np["ti"][i]
    nusc_map = find_map(batch_np["traj_i"][i], nusc, nusc_map_d)
    nusc_map.explorer.color_map["lane"] = "#FFFFFF"

    bs = batch_np["traj_i"].shape[0]

    action_label = {0:"keep", 1:"left-lane-change", 2:"right-lane-change", 3:"outlier"}

    # plot hd map
    ego_trajs = batch_np["ego_traj"][i]
    ego_xy, ego_th, ego_v, ego_L, ego_W = ego_trajs[0, :2], ego_trajs[0, 2], ego_trajs[0, 3], ego_trajs[0, 4], ego_trajs[0, 5]
    r = 50
    my_patch = (ego_xy[0]-r,  ego_xy[1]-r, ego_xy[0]+r, ego_xy[1]+r)
    fig, ax = nusc_map.render_map_patch(my_patch, [xx for xx in nusc_map.non_geometric_layers if xx not in ['traffic_light', 'walkway', "ped_crossing", "stop_line"]], 
                        alpha=0.3, figsize=(8, 8), bitmap=None)                
    bev_handles, bev_labels = ax.get_legend_handles_labels()

    # plot all neighbors
    sample_token = dataloader.dataset.meta_d[traj_i][ti]
    my_sample = nusc.get("sample", sample_token)
    for ann_token in my_sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        instance = nusc.get("instance", ann["instance_token"])
        category = nusc.get("category", instance["category_token"])
        if "vehicle" in category["name"]:
            other_rot = ann["rotation"]
            other_pose = ann["translation"]
            other_th = get_th_from_rotation(ann["rotation"])
            other_L = ann["size"][1]
            other_W = ann["size"][0]
            plot_agent((other_pose[0], other_pose[1]), other_th, other_L, other_W, ax, color="gray", alpha=0.5)

    # plot focus neighbors
    neighbors = batch_np["neighbors"][i]
    for ii in range(neighbors.shape[0]):
        if neighbors[ii, 0] == 1:
            nei = neighbors[ii, 1:]
            plot_agent((nei[0], nei[1]), nei[2], nei[4] * 1.2, nei[5] * 1.2, ax, color="brown", alpha=0.3)

    # plot current centerlines
    currlane = batch_np["currlane_wpts"][i].reshape((-1, 3))
    leftlane = batch_np["leftlane_wpts"][i].reshape((-1, 3))
    rightlane = batch_np["rightlane_wpts"][i].reshape((-1, 3))        
    plt.plot(currlane[:, 0], currlane[:, 1], "blue", linewidth=6, alpha=0.4, label="currlane")
    plt.plot(leftlane[:, 0], leftlane[:, 1], "green", linewidth=6, alpha=0.4, label="leftlane")
    plt.plot(rightlane[:, 0], rightlane[:, 1], "red", linewidth=6, alpha=0.4, label="rightlane")

    lanes_d = {0:currlane, 1:leftlane, 2:rightlane}

    color_list=["blue", "green", "red"]

    # plot ego, ego trajs + nn trajs
    if multi_check:
        dense_trajs_np = batch_np["dense_trajs"].reshape((-1, args.n_randoms, 3)+batch_np["dense_trajs"].shape[-2:])[i]
        plot_agent(ego_xy, ego_th, ego_L, ego_W, ax, color="blue")
        high_level = batch_np["gt_high_level"][i, 0]

        # print("VIZ %02d ego_xy:%.3f %.3f nn_xy:%.3f %.3f dense_xy:%.3f %.3f"%(i, ego_xy[0], ego_xy[1], ))
        plt.plot(ego_trajs[:, 0], ego_trajs[:, 1], color="cyan", alpha=0.8, linewidth=2.5, zorder=500, label="gt_traj")
        
        # plot multiple ego trajs
        assert not (ego_only and opt_only)
        if ego_only or not opt_only:
            nn_trajs_np = batch_np["nn_trajs"].reshape((-1, args.n_randoms, 3)+batch_np["nn_trajs"].shape[-2:])[i]
            for ii in range(args.n_randoms):
                for kk in range(3):
                    if lanes_d[kk][0,0]!=0:
                        if args.viz_correct and viz_cache[split]["scores_all"].reshape(bs, args.n_randoms, 3)[i,ii,kk]<=0:
                            continue
                        plt.plot(nn_trajs_np[ii, kk, :, 0], nn_trajs_np[ii, kk, :, 1], color=color_list[kk], alpha=0.8, linewidth=1, zorder=800, label="diffusion (mode=%d)"%(kk) if ii==0 else None)
        
        # plot multiple trajopt trajs
        if opt_only or not ego_only:
            for ii in range(args.n_randoms):
                for kk in range(3):
                    if lanes_d[kk][0,0]!=0:
                        if args.viz_correct and viz_cache[split]["dense_scores"].reshape(bs, args.n_randoms, 3)[i,ii,kk]<=0:
                            continue
                        plt.plot(dense_trajs_np[ii, kk,:, 0], dense_trajs_np[ii, kk, :, 1], color=color_list[kk], alpha=0.7, linewidth=1, zorder=800, label="trajopt (mode=%d)"%(kk) if ii==0 else None)
    else:
        nn_trajs_np = batch_np["nn_trajs"][i]
        for ii in range(args.n_randoms):
            kk = int(viz_cache[split]["gt_high_level"][i,0].item())
            if kk==-1 or kk==3:
                kk=0
            if args.viz_correct and viz_cache[split]["scores_all"].reshape(bs, args.n_randoms)[i,ii]<=0:
                continue
            plt.plot(nn_trajs_np[ii, :, 0], nn_trajs_np[ii, :, 1], color=color_list[kk], alpha=0.8, linewidth=1, zorder=800, label="mono (mode=%d)"%(kk) if ii==0 else None)
        plt.plot(ego_trajs[:, 0], ego_trajs[:, 1], color="cyan", alpha=0.8, linewidth=2.5, zorder=500, label="gt_traj")
    ax.legend(frameon=True, loc='upper right')

    # plot high_level actions
    if opt_only:
        plt.title("traj:%d t:%d gt_high_level:%s"%(
            batch_np["traj_i"][i], batch_np["ti"][i], action_label[batch_np["gt_high_level"][i, 0]],
            ))
    else:
        if multi_check:
            plt.title("traj:%d ti:%d lbl:%s s:(%.2f|%.2f %.2f %.2f) acc(%.2f|%.2f %.2f %.2f)/gt %.2f"%(
                batch_np["traj_i"][i], batch_np["ti"][i], action_label[batch_np["gt_high_level"][i, 0]],
                np.mean(batch_np["scores_all"].reshape(bs, args.n_randoms, 3)[i]), 
                np.mean(batch_np["scores_all"].reshape(bs, args.n_randoms, 3)[i, :, 0]), 
                np.mean(batch_np["scores_all"].reshape(bs, args.n_randoms, 3)[i, :, 1]), 
                np.mean(batch_np["scores_all"].reshape(bs, args.n_randoms, 3)[i, :, 2]), 
                np.mean(batch_np["scores_all"].reshape(bs, args.n_randoms, 3)[i]>0), 
                np.mean(batch_np["scores_all"].reshape(bs, args.n_randoms, 3)[i, :, 0]>0), 
                np.mean(batch_np["scores_all"].reshape(bs, args.n_randoms, 3)[i, :, 1]>0), 
                np.mean(batch_np["scores_all"].reshape(bs, args.n_randoms, 3)[i, :, 2]>0), 
                batch_np["scores_gt_all"][i],
                ))
        else:
            plt.title("traj:%d ti:%d lbl:%s acc:%.2f gt:%.2f score:%.2f"%(
                batch_np["traj_i"][i], batch_np["ti"][i], action_label[batch_np["gt_high_level"][i, 0]],
                np.mean(batch_np["scores_all"].reshape(bs, args.n_randoms)[i]>0), 
                np.mean(batch_np["scores_gt_all"].reshape(bs, 1)[i]>0), 
                batch_np["scores_gt_all"][i],
            ))
    plt.axis("scaled")
    x_min, y_min, x_max, y_max = my_patch
    x_margin = np.minimum(x_max - x_min / 4, 50)
    y_margin = np.minimum(y_max - y_min / 4, 10)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    if multi_check:
        if ego_only:
            plt.savefig("%s/viz_e%03d_%s_tr%03d_i%03d_diffusion.png"%(args.viz_dir, epi, split, batch_np["traj_i"][i], batch_np["ti"][i]), bbox_inches='tight', pad_inches=0.1)
        elif opt_only:
            plt.savefig("%s/viz_e%03d_%s_tr%03d_i%03d_trajopt.png"%(args.viz_dir, epi, split, batch_np["traj_i"][i], batch_np["ti"][i]), bbox_inches='tight', pad_inches=0.1)
        else:
            plt.savefig("%s/viz_e%03d_%s_tr%03d_i%03d.png"%(args.viz_dir, epi, split, batch_np["traj_i"][i], batch_np["ti"][i]), bbox_inches='tight', pad_inches=0.1)
    else:
        plt.savefig("%s/viz_e%03d_%s_tr%03d_i%03d.png"%(args.viz_dir, epi, split, batch_np["traj_i"][i], batch_np["ti"][i]), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_a_single_scene(nusc, nusc_map_d, traj_i, ti, sample_token, ego_trajs,
                    neighbors, currlane_wpts, leftlane_wpts, rightlane_wpts,img_path):
    nusc_map = find_map(traj_i, nusc, nusc_map_d)
    nusc_map.explorer.color_map["lane"] = "#FFFFFF"
    ego_trajs = to_np(ego_trajs)
    ego_xy, ego_th, ego_v, ego_L, ego_W = ego_trajs[0, :2], ego_trajs[0, 2], ego_trajs[0, 3], ego_trajs[0, 4], ego_trajs[0, 5]
    r = 50
    my_patch = (ego_xy[0]-r,  ego_xy[1]-r, ego_xy[0]+r, ego_xy[1]+r)
    fig, ax = nusc_map.render_map_patch(my_patch, 
        [xx for xx in nusc_map.non_geometric_layers if xx not in ['traffic_light', 'walkway', "ped_crossing", "stop_line"]], 
            alpha=0.3, figsize=(8, 8), bitmap=None)      
    bev_handles, bev_labels = ax.get_legend_handles_labels()
    my_sample = nusc.get("sample", sample_token)
    for ann_token in my_sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        instance = nusc.get("instance", ann["instance_token"])
        category = nusc.get("category", instance["category_token"])
        if "vehicle" in category["name"]:
            other_rot = ann["rotation"]
            other_pose = ann["translation"]
            other_th = get_th_from_rotation(ann["rotation"])
            other_L = ann["size"][1]
            other_W = ann["size"][0]
            plot_agent((other_pose[0], other_pose[1]), other_th, other_L, other_W, ax, color="gray", alpha=0.5)
    # plot focus neighbors
    neighbors = to_np(neighbors)
    for ii in range(neighbors.shape[0]):
        if neighbors[ii, 0] == 1:
            nei = neighbors[ii, 1:]
            plot_agent((nei[0], nei[1]), nei[2], nei[4] * 1.2, nei[5] * 1.2, ax, color="brown", alpha=0.3)
    # plot current centerlines
    currlane = to_np(currlane_wpts).reshape((-1, 3))
    leftlane = to_np(leftlane_wpts).reshape((-1, 3))
    rightlane = to_np(rightlane_wpts).reshape((-1, 3))       
    plt.plot(currlane[:, 0], currlane[:, 1], "blue", linewidth=6, alpha=0.4, label="currlane")
    plt.plot(leftlane[:, 0], leftlane[:, 1], "green", linewidth=6, alpha=0.4, label="leftlane")
    plt.plot(rightlane[:, 0], rightlane[:, 1], "red", linewidth=6, alpha=0.4, label="rightlane")
    plot_agent(ego_xy, ego_th, ego_L, ego_W, ax, color="blue")
    plt.plot(ego_trajs[:, 0], ego_trajs[:, 1], color="cyan", alpha=0.8, linewidth=2.5, zorder=500, label="gt_traj")
    ax.legend(frameon=True, loc='upper right')
    plt.axis("scaled")
    x_min, y_min, x_max, y_max = my_patch
    x_margin = np.minimum(x_max - x_min / 4, 50)
    y_margin = np.minimum(y_max - y_min / 4, 10)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    utils.plt_save_close(img_path)


def plot_debug_scene(batch_np, dense_trajs, dense_scores, nn_trajs, scores_all, args, iter_i, i=None, tj_n_randoms=None, nn_n_randoms=None, extra_i=None):
    if i is None:
        i = 0
    if tj_n_randoms is None:
        tj_n_randoms = args.n_randoms
    if nn_n_randoms is None:
        nn_n_randoms = args.n_randoms

    plt.figure(figsize=(16, 10))
    ego_trajs = batch_np["ego_traj"][i]
    ego_xy, ego_th, ego_v, ego_L, ego_W = ego_trajs[0, :2], ego_trajs[0, 2], ego_trajs[0, 3], ego_trajs[0, 4], ego_trajs[0, 5]
    r = 50
    my_patch = (ego_xy[0]-r,  ego_xy[1]-r, ego_xy[0]+r, ego_xy[1]+r)

    color1_list=["blue", "green", "red"]
    num_subs = 2
    for sub_i in range(num_subs):
        plt.subplot(1, num_subs, sub_i+1)
        ax = plt.gca()

        # plot lanes
        currlane = batch_np["currlane_wpts"][i].reshape((-1, 3))
        leftlane = batch_np["leftlane_wpts"][i].reshape((-1, 3))
        rightlane = batch_np["rightlane_wpts"][i].reshape((-1, 3))        
        plt.plot(currlane[:, 0], currlane[:, 1], "blue", linewidth=6, alpha=0.4, label="currlane")
        plt.plot(leftlane[:, 0], leftlane[:, 1], "green", linewidth=6, alpha=0.4, label="leftlane")
        plt.plot(rightlane[:, 0], rightlane[:, 1], "red", linewidth=6, alpha=0.4, label="rightlane")

        # plot focus neighbors
        neighbors = batch_np["neighbors"][i]
        for ii in range(neighbors.shape[0]):
            if neighbors[ii, 0] == 1:
                nei = neighbors[ii, 1:]
                plot_agent((nei[0], nei[1]), nei[2], nei[4] * 1.2, nei[5] * 1.2, ax, color="brown", alpha=0.3)

        # plot different trajs
        lanes_d = {0:currlane, 1:leftlane, 2:rightlane}
        
        plt.plot(ego_trajs[:, 0], ego_trajs[:, 1], color="cyan", alpha=0.8, linewidth=4, zorder=500, label="gt_traj")

        plot_agent(ego_xy, ego_th, ego_L, ego_W, ax, color="blue")
        n_randoms = tj_n_randoms if sub_i==0 else nn_n_randoms
        if sub_i==0:
            the_trajs = dense_trajs[i,:,:,:-1,:]
            the_scores = dense_scores.reshape(-1, n_randoms, 3)
            the_label_str = "TrajOpt (mode=%d)"
        elif sub_i==1:
            the_trajs = nn_trajs[..., :-1, :].reshape((-1, n_randoms, 3, args.nt, nn_trajs.shape[-1]))[i]
            the_scores = scores_all.reshape(-1, n_randoms, 3)
            the_label_str = "Diffusion (mode=%d)"
        else:
            nn_trajs_zero = batch_np["nn_trajs_zero"]
            the_trajs = nn_trajs_zero[..., :-1, :].reshape((-1, n_randoms, 3, args.nt, nn_trajs.shape[-1]))[i]
            the_scores = scores_all.reshape(-1, n_randoms, 3)
            the_label_str = "DiffusionZero (mode=%d)"

        for ii in range(n_randoms):
            for kk in range(3):
                if lanes_d[kk][0,0]!=0:
                    color = color1_list[kk]
                    if args.sampling_size < 7 and sub_i!=0:
                        if kk==0:
                            color_list_list=["powderblue", "lightskyblue", "cadetblue", "dodgerblue", "royalblue", "navy"]
                        elif kk==1:
                            color_list_list=["greenyellow", "lightgreen", "limegreen", "forestgreen", "seagreen", "darkseagreen"]
                        else:
                            color_list_list=["rosybrown", "lightcoral", "brown", "coral", "firebrick", "maroon"]
                        color = color_list_list[ii]

                    if args.viz_correct and the_scores[i,ii,kk]<=0:
                        continue
                    plt.plot(the_trajs[ii, kk, :, 0], the_trajs[ii, kk, :, 1], 
                                color=color, alpha=0.8, linewidth=1, zorder=800, 
                                label=the_label_str%(kk) if ii==0 else None)
        plt.axis("scaled")
        x_min, y_min, x_max, y_max = my_patch
        x_margin = np.minimum(x_max - x_min / 4, 50)
        y_margin = np.minimum(y_max - y_min / 4, 10)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.legend(frameon=True, loc='upper right')

    # save
    if extra_i is not None:
        utils.plt_save_close("%s/viz_it%06d_tr%03d_i%03d_dfs%03d.png"%(args.viz_dir, iter_i, batch_np["traj_i"][i], batch_np["ti"][i], extra_i))
    else:
        utils.plt_save_close("%s/viz_it%06d_tr%03d_i%03d.png"%(args.viz_dir, iter_i, batch_np["traj_i"][i], batch_np["ti"][i]))