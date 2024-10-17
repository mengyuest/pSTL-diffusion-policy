from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
import os
import time
import pickle
import os.path as osp
import numpy as np
import torch
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap
from utils import euler_from_quaternion, compute_entropy, to_np
from scipy.spatial import ConvexHull

class NuscenesPkl(NuScenes):
    def __init__(self, version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        pkl_file_path = osp.join(self.table_root, 'all_data.pickle')
        if osp.exists(pkl_file_path)==False:
            # Explicitly assign tables to help the IDE determine valid class members.
            self.category = self.__load_table__('category')
            self.attribute = self.__load_table__('attribute')
            self.visibility = self.__load_table__('visibility')
            self.instance = self.__load_table__('instance')
            self.sensor = self.__load_table__('sensor')
            self.calibrated_sensor = self.__load_table__('calibrated_sensor')
            self.ego_pose = self.__load_table__('ego_pose')
            self.log = self.__load_table__('log')
            self.scene = self.__load_table__('scene')
            self.sample = self.__load_table__('sample')
            self.sample_data = self.__load_table__('sample_data')
            self.sample_annotation = self.__load_table__('sample_annotation')
            self.map = self.__load_table__('map')

            # Initialize the colormap which maps from class names to RGB values.
            self.colormap = get_colormap()

            # Initialize map mask for each map record.
            for map_record in self.map:
                map_record['mask'] = MapMask(osp.join(self.dataroot, map_record['filename']), resolution=map_resolution)

            if verbose:
                for table in self.table_names:
                    print("{} {},".format(len(getattr(self, table)), table))
                print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

            # Make reverse indexes for common lookups.
            self.__make_reverse_index__(verbose)

            # TODO(yue)
            all_dict = {}
            for table in self.table_names + ["_token2ind"]:
                all_dict[table] = getattr(self, table)
            with open(pkl_file_path, "wb") as ff:
                pickle.dump(all_dict, ff, pickle.HIGHEST_PROTOCOL)

        else:
            with open(pkl_file_path, "rb") as ff:
                all_dict = pickle.load(ff)
            for table in self.table_names + ["_token2ind"]:
                setattr(self, table, all_dict[table])
            self.colormap = get_colormap()
            if verbose:
                for table in self.table_names:
                    print("{} {},".format(len(getattr(self, table)), table))
                print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Initialize NuScenesExplorer class.
        self.explorer = NuScenesExplorer(self)

def get_nuscenes(is_mini=False, queue=None):
    dataroot=os.environ["MY_DATA_DIR"]
    if len(dataroot)<1:
        exit("CANNOT FIND ENV VARIABLE:%s"%(dataroot))
    else:
        dataroot=os.path.join(dataroot, "nuscenes")
    if is_mini:
        nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
    else:
        nusc = NuscenesPkl(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    nusc_map_d={k: NuScenesMap(dataroot=dataroot, map_name=k) for k in \
        ["boston-seaport", "singapore-hollandvillage", "singapore-queenstown", "singapore-onenorth"]}
    if queue is not None:
        queue.put((nusc, nusc_map_d))
        return
    else:
        return nusc, nusc_map_d

# get all the sample_tokens for each scene,
# from all the scenes in the nusc dataset
def get_scene_tokens(nusc):
    meta_list = []
    for i, scene in enumerate(nusc.scene):
        tokens=[]
        sample_token = scene["first_sample_token"]
        while sample_token != "":
            tokens.append(sample_token)
            sample_token = nusc.get("sample", sample_token)["next"]
        meta_list.append((i, tokens))
    return meta_list

# get ego pose
# the speed v_t is estimated from s_t, and s_{t+1}
# the last speed v_t will be v_{t-1} + (v_{t-1} - v_{t-2})/dt * dt
def get_ego_trajectory(nusc, tokens, dt, return_numpy=False):
    ego_trajs = []
    for ti, token in enumerate(tokens):
        sample = nusc.get("sample", token)
        ego_state = get_ego_state_from_sample(nusc, sample)
        ego_trajs.append(ego_state)
    ego_trajs = torch.stack(ego_trajs, dim=0)
    # add the velocity estimates
    v = torch.norm(ego_trajs[1:, :2]-ego_trajs[:-1, :2], dim=-1) / dt
    if v.shape[0]==1:
        v = torch.cat([v, v], dim=0)
    else:
        v_last = v[-1:] * 2 - v[-2:-1]
        v = torch.cat([v, v_last], dim=0)
    ego_trajs = torch.cat([ego_trajs[:, :3], v.unsqueeze(-1), ego_trajs[:, 3:]], dim=-1) 
    if return_numpy:
        return ego_trajs.numpy()
    else:
        return ego_trajs

# get neighbors
# TODO: for now, only focus on current frame, 
# and constant speed neighbors prediction
def get_neighbors(nusc, token, ret_full=False):
    sample = nusc.get("sample", token)
    neighbors = []
    ann_tokens = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if "vehicle" in ann["category_name"]:
            nei_state = get_nei_state_from_annotation(nusc, ann)
            neighbors.append(nei_state)
            ann_tokens.append(ann_token)
    if ret_full:
        return neighbors, ann_tokens
    else:
        return neighbors

# used only for nuscenes record
# they seems to have a weird format for heading angle def
# numpy input/output
def get_th_from_rotation(rotation):
    return np.pi - euler_from_quaternion(rotation)[0]

# get the difference measure for two angles in [-inf, inf]
# input numpy, two angles
# output range [0, 1]
# 0 ~ similar, or 2k pi
# 1 ~ largest difference (2k+1)pi diff
def angle_diff(a, b):
    return 1/2*(1-np.cos(a-b))

# compute the trajectory length
# input: (n, 2+n>=0)
# return: 1
def compute_traj_len(trajs):
    trajs_np = np.array(trajs)
    return np.sum(np.linalg.norm(trajs_np[1:, :2] - trajs_np[:-1, :2], axis=-1))

def get_traj_len_np(traj: np.ndarray):
    return np.sum(np.linalg.norm(traj[1:,:2]-traj[:-1,:2], axis=-1))

#################
# important API #
#################
def get_closest_centerlane_with_heuristcs(nusc, nusc_map, x, y, radius, trajs, n_expands, n_segs, lanes_cut=False):
    lanes = nusc_map.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    discrete_points = nusc_map.discretize_lanes(lanes, 0.5)
    current_min = np.inf
    min_check_head_dist=None
    min_id = ""
    min_state_dict_i = None

    # first find the closed one based on metrics
    for lane_id, points_3d in discrete_points.items():
        points_3d_np = np.array(points_3d)
        points = np.array(points_3d)[:, :2]
        state_dist_all = np.linalg.norm(points - [x, y], axis=1).min()
        state_dist = state_dist_all.min()
        state_dist_i = state_dist_all.argmin()

        traj_dist = np.linalg.norm(points[None, :] - trajs[:5, None, :2], axis=2).min(axis=1).mean()
        head_dist = angle_diff(trajs[0, 2], points_3d_np[state_dist_i, 2])
        total_score = state_dist + traj_dist + head_dist
        if total_score < current_min:
            current_min = total_score
            min_id = lane_id
            min_check_head_dist = head_dist
            min_poses = points_3d_np
            min_state_dict_i = state_dist_i    

    # remove the case when the heading is very far from the lane direction
    if min_id != "" and min_check_head_dist > 0.8:
        min_id = ""
        
    # then expand a bit
    if min_id != "":
        # make sure the starting point is very close to the ego vehicle
        if lanes_cut:
            min_poses = min_poses[max(0, min_state_dict_i-5):]

        total_traj_len = compute_traj_len(trajs)
        curr_lane_len = compute_traj_len(min_poses)
        lane_id_list = [min_id]
        lane_poses_list = [min_poses]
        for expand_i in range(n_expands):
            lane_id = lane_id_list[-1]
            lane_poses = lane_poses_list[-1]
            outs = nusc_map.get_outgoing_lane_ids(lane_id)
            next_ones = []
            for out_id in outs:
                arcline_path = nusc_map.arcline_path_3.get(out_id)
                if arcline_path: 
                    out_record = arcline_path
                    out_poses = arcline_path_utils.discretize_lane(out_record, resolution_meters=0.5)
                    dist_error = np.linalg.norm(np.array(out_poses[0][:2])-np.array(lane_poses[-1][:2]))
                    head_error = angle_diff(out_poses[0][2], lane_poses[-1][2])
                    next_ones.append((dist_error + head_error, (out_id, out_poses)))

            if len(next_ones)==0:
                break
            next_ones = sorted(next_ones, key = lambda x:x[0])
            nextlane_id, nextlane_poses = next_ones[0][1]
            lane_id_list.append(nextlane_id)
            lane_poses_list.append(nextlane_poses)
            curr_lane_len += compute_traj_len(nextlane_poses)
            if curr_lane_len > total_traj_len + 10:
                break
        
        lane_poses_list = [np.array(lane_poses) for lane_poses in lane_poses_list]
        centerlane_poses_full = np.concatenate(lane_poses_list, axis=0)
        numElems = n_segs
        sub_poses_list = []
        for lane_poses in lane_poses_list:
            evenspace_idx = np.round(np.linspace(0, lane_poses.shape[0] - 1, numElems)).astype(int)
            sub_poses = lane_poses[evenspace_idx]
            sub_poses_list.append(sub_poses)
        sub_poses_full = np.concatenate(sub_poses_list, axis=0)
        evenspace_idx = np.round(np.linspace(0, sub_poses_full.shape[0] - 1, numElems)).astype(int)
        centerlane_poses = sub_poses_full[evenspace_idx]
    
    else:
        centerlane_poses = np.zeros((n_segs, 2))
        min_poses = np.zeros((n_segs, 2))
    return min_id, centerlane_poses, min_poses


def is_able_uturn(nusc_map, ego_state, currlane_wpts, testlane_wpts):
    valid_uturn=False
    if np.cos(testlane_wpts[0, -1] - currlane_wpts[0, -1]) < -0.9: # (this almost opposite)
        valid_uturn=True

        # checking speed
        if ego_state[3] > 3:
            valid_uturn = False

        # checking lane-to-lane distance
        dist = np.linalg.norm(testlane_wpts[0, :2]-currlane_wpts[0, :2])
        if dist > 8:
            valid_uturn = False

        # checking whether there is hole in front
        # between curr lane and adj lane
        # in front 4, 5, 6, 7, 8 meters
        # if no whole, then avail to u-turn
        mid = (testlane_wpts[0, :2] + currlane_wpts[0, :2])/2
        theta = currlane_wpts[0, 2]
        for test_d in [4, 6, 8]:
            test_point = np.array([mid[0] + test_d * np.cos(theta), mid[1] + test_d * np.sin(theta)])
            if check_drive_availability(nusc_map, test_point)==False:
                # print("there is hole at", test_point)
                valid_uturn = False
                break
    return valid_uturn

def check_drive_availability(nusc_map, point):
    layers = nusc_map.explorer.layers_on_point(point[0], point[1])
    return layers["drivable_area"]!=""

def find_start_and_end_lane_idx_for_traj(lane, traj, minimum_consider_len):
    # lane (M, 3)
    # traj (N, 2)
    t2l_dist = np.linalg.norm(lane[:, :2] - traj[0:1, :2], axis=-1)
    begin_idx = np.argmin(t2l_dist)

    compared_len = max(minimum_consider_len, get_traj_len_np(traj))
    frag_length = np.linalg.norm(lane[1:,:2]-lane[:-1,:2], axis=-1)  # (n-1, )
    frag_length = np.concatenate((np.zeros((1,)), frag_length))
    frag_cumsum = np.cumsum(frag_length)
    if np.all(frag_cumsum - frag_cumsum[begin_idx]<compared_len):
        end_idx = lane.shape[0]-1
    else:
        end_idx = np.argmax(frag_cumsum - frag_cumsum[begin_idx]>=compared_len)
    info = {"t2l_dist": t2l_dist}
    return begin_idx, end_idx, info

def get_centerline_from_anno(nusc: NuScenes, nusc_map: NuScenesMap, keyframe, lane_key, radius, ego_trajs, n_expands, n_segs, highlevel):
    DIST_THRES = 7.0
    LANE_WIDTH = 4.0
    MINIMUM_CONSIDER_LEN = 20.0
    records = keyframe["lanes"][lane_key]
    pts_anno = [pts for li, (_, _, pts) in enumerate(records)]
    ids_anno = [li * np.ones_like(pts[:,0]) for li, (_, _, pts) in enumerate(records)]
    if len(pts_anno)!=0:
        pts_anno = np.concatenate(pts_anno, axis=0)
        ids_anno = np.concatenate(ids_anno, axis=0)
        compared_len = max(MINIMUM_CONSIDER_LEN, get_traj_len_np(ego_trajs))
        begin_idx, end_idx, info = find_start_and_end_lane_idx_for_traj(pts_anno, ego_trajs, MINIMUM_CONSIDER_LEN)

        min_id = records[int(ids_anno[begin_idx])][1]
        lane_full = pts_anno[begin_idx:end_idx+1]
        
        evenspace_idx = np.round(np.linspace(0, lane_full.shape[0] - 1, n_segs)).astype(int)
        lane_wpts = lane_full[evenspace_idx]
    else:
        min_id = ''
        lane_full = lane_wpts = np.zeros((n_segs, 3))

    if lane_key=="curr":  # the current solution might be too short for some case at the end of the trajectories
        curr_lane_len = get_traj_len_np(lane_full) if min_id!="" else MINIMUM_CONSIDER_LEN+1

        if curr_lane_len < MINIMUM_CONSIDER_LEN:
            lane_full = pts_anno[begin_idx:]

            total_traj_len = MINIMUM_CONSIDER_LEN

            lane_id_list = [min_id]
            lane_poses_list = [lane_full]
            for expand_i in range(n_expands):
                lane_id = lane_id_list[-1]
                lane_poses = lane_poses_list[-1]
                outs = nusc_map.get_outgoing_lane_ids(lane_id)
                next_ones = []
                for out_id in outs:
                    arcline_path = nusc_map.arcline_path_3.get(out_id)
                    if arcline_path:
                        out_poses = arcline_path_utils.discretize_lane(arcline_path, resolution_meters=0.5)
                        dist_error = np.linalg.norm(np.array(out_poses[0][:2])-np.array(lane_poses[-1][:2]))
                        head_error = angle_diff(out_poses[0][2], lane_poses[-1][2])
                        next_ones.append((dist_error + head_error, (out_id, np.array(out_poses))))
                if len(next_ones)==0:
                    break
                next_ones = sorted(next_ones, key = lambda x:x[0])
                nextlane_id, nextlane_poses = next_ones[0][1]
                lane_id_list.append(nextlane_id)
                lane_poses_list.append(nextlane_poses)
                curr_lane_len += get_traj_len_np(nextlane_poses)
                if curr_lane_len > total_traj_len:
                    break
            lane_full = np.concatenate(lane_poses_list, axis=0)
            evenspace_idx = np.round(np.linspace(0, lane_full.shape[0] - 1, n_segs)).astype(int)
            lane_wpts = lane_full[evenspace_idx]
            curr_lane_len = get_traj_len_np(lane_full)
    else:
        # first make sure not in the intersection
        x, y, th = ego_trajs[0, 0:3]        
        layers = nusc_map.explorer.layers_on_point(x, y)
        if (lane_key=='left' and highlevel==1) or (lane_key=='right' and highlevel==2):
            is_intersection = False
        else:
            segment_token = layers['road_segment']
            if segment_token=="":
                is_intersection = False
            else:
                is_intersection = nusc_map.get("road_segment", segment_token)['is_intersection'] 
        if is_intersection:
            min_id = ""
            lane_full = lane_wpts = np.zeros((n_segs, 3))
        prev_min_id = min_id
        min_pts_lane_dist = info["t2l_dist"][begin_idx] if prev_min_id!='' else DIST_THRES
        if min_pts_lane_dist >= DIST_THRES:  # if too far from the left/right centerline
            # should find new centerline
            min_id = ""
            lane_full = lane_wpts = np.zeros((n_segs, 3))
            if not is_intersection:
                # find closest left/right 
                # using the old method to find one
                if lane_key == "left":
                    new_x = x + LANE_WIDTH * np.cos(th+np.pi/2)
                    new_y = y + LANE_WIDTH * np.sin(th+np.pi/2)
                else:
                    new_x = x + LANE_WIDTH * np.cos(th-np.pi/2)
                    new_y = y + LANE_WIDTH * np.sin(th-np.pi/2)
                lanes = nusc_map.get_records_in_radius(new_x, new_y, radius, ['lane', 'lane_connector'])
                lanes = lanes['lane'] + lanes['lane_connector']
                discrete_points = nusc_map.discretize_lanes(lanes, 0.5)

                # first find the closed one based on metrics
                current_min=min_pts_lane_dist
                for lane_id, points_3d in discrete_points.items():
                    if lane_id != prev_min_id: # should not be the same as the previous lane
                        points_3d_np = np.array(points_3d)
                        points = points_3d_np[:, :2]
                        state_dist_all = np.linalg.norm(points - [new_x, new_y], axis=1)
                        state_dist_i = state_dist_all.argmin()
                        state_dist = state_dist_all[state_dist_i]
                        total_score = state_dist
                        if total_score < current_min:
                            current_min = total_score
                            min_id = lane_id
                            min_poses = points_3d_np
                            min_state_dict_i = state_dist_i    

                if min_id != "":
                    # make sure the starting point is very close to the ego vehicle
                    # and make sure the lane is different from the current lane
                    min_poses = min_poses[min_state_dict_i:]
                    total_traj_len = get_traj_len_np(ego_trajs)
                    curr_lane_len = get_traj_len_np(min_poses)
                    lane_id_list = [min_id]
                    lane_poses_list = [min_poses]
                    if curr_lane_len > total_traj_len:
                        begin_idx, end_idx, info = find_start_and_end_lane_idx_for_traj(min_poses, ego_trajs, MINIMUM_CONSIDER_LEN)
                        lane_poses_list = [min_poses[begin_idx:end_idx+1]] 
                    else:
                        for expand_i in range(n_expands):
                            lane_id = lane_id_list[-1]
                            lane_poses = lane_poses_list[-1]
                            outs = nusc_map.get_outgoing_lane_ids(lane_id)
                            next_ones = []
                            for out_id in outs:
                                arcline_path = nusc_map.arcline_path_3.get(out_id)
                                if arcline_path:
                                    out_poses = arcline_path_utils.discretize_lane(arcline_path, resolution_meters=0.5)
                                    dist_error = np.linalg.norm(np.array(out_poses[0][:2])-np.array(lane_poses[-1][:2]))
                                    head_error = angle_diff(out_poses[0][2], lane_poses[-1][2])
                                    next_ones.append((dist_error + head_error, (out_id, np.array(out_poses))))
                            if len(next_ones)==0:
                                break
                            next_ones = sorted(next_ones, key = lambda x:x[0])
                            nextlane_id, nextlane_poses = next_ones[0][1]
                            lane_id_list.append(nextlane_id)
                            lane_poses_list.append(nextlane_poses)
                            curr_lane_len += get_traj_len_np(nextlane_poses)
                            if curr_lane_len > total_traj_len:
                                break
                    lane_full = np.concatenate(lane_poses_list, axis=0)
                    evenspace_idx = np.round(np.linspace(0, lane_full.shape[0] - 1, n_segs)).astype(int)
                    lane_wpts = lane_full[evenspace_idx]
    if min_id == "":
        min_id = -1
    return min_id, lane_wpts, lane_full


# compute the distance of two trajectories
# input: (n1, 2), (n2, 2)
# return: ()
def traj_diff(lane_a, lane_b):
    a = np.array(lane_a)[:, :2]
    b = np.array(lane_b)[:, :2]
    d = np.linalg.norm(a[None]-b[:,None], axis=-1)
    matched_ratio = np.mean(np.min(d,axis=0) < 2.0)
    return matched_ratio


def compute_traj_diff(lane_a, lane_b):
    a = np.array(lane_a)[:, :2]
    b = np.array(lane_b)[:, :2]
    d = np.linalg.norm(a[None]-b[:,None], axis=-1)
    dist_a_to_b = np.mean(np.min(d,axis=0))
    dist_b_to_a = np.mean(np.min(d,axis=1))
    return min(dist_b_to_a, dist_a_to_b)

# get centerlines
# select to use annotation, or pure heuristic, or simplified lane model, or other paper used methods
# make this adapt to the nuscene-gui
def get_centerlines(nusc, nusc_map, token, ti, ego_trajs, anno_data, n_expands, n_segs, lanes_cut=False, ret_full=False, highlevel=0):
    radius = 2
    LANE_WIDTH = 4.0
    ego_state = ego_trajs[0]
    x, y, ego_th = ego_state[0:3]
    if len(anno_data)>1:
        keys = sorted(anno_data)
        for k_i, key in enumerate(keys):
            if ti>=key and (k_i==len(keys)-1 or ti<keys[k_i+1]):
                break
        keyframe = anno_data[key]
    else:
        keyframe = anno_data[0]
    curr_id, currlane_wpts, currlane_full = get_centerline_from_anno(nusc, nusc_map, keyframe, "curr", radius, ego_trajs, n_expands, n_segs, highlevel=highlevel)
    left_id, leftlane_wpts, leftlane_full = get_centerline_from_anno(nusc, nusc_map, keyframe, "left", radius, ego_trajs, n_expands, n_segs, highlevel=highlevel)
    right_id, rightlane_wpts, rightlane_full = get_centerline_from_anno(nusc, nusc_map, keyframe, "right", radius, ego_trajs, n_expands, n_segs, highlevel=highlevel)

    if left_id == curr_id or compute_traj_diff(currlane_full, leftlane_full) < 0.5:
        leftlane_wpts = leftlane_full = leftlane_wpts * 0
        left_id = -1
    if right_id == curr_id or compute_traj_diff(currlane_full, rightlane_full) < 0.5:
        rightlane_wpts = rightlane_full = rightlane_wpts * 0
        right_id = -1
   
    # TODO(yue)
    # each returns a full lengths of the centerline (len(road), 3)
    # and then finegrained/fixed-length roads is handled elsewhere
    if ret_full:
        return curr_id, currlane_wpts, currlane_full, \
               left_id, leftlane_wpts, leftlane_full, \
               right_id, rightlane_wpts,rightlane_full 
    else:
        return currlane_wpts, leftlane_wpts, rightlane_wpts

# get high-level behaviors
# TODO this way of getting high-level behaviors need to be re-examined
def get_high_level_behaviors(nusc, anno_data, ti, nt, buffer, ego_traj_np):
    inv_d = {"Lane-keeping":0., None:0., "Left-lane-change":1., "Right-lane-change":2., "Stop sign":3., "Traffic light":3.}
    if len(anno_data)==1:
        gt_high_level = inv_d[anno_data[0]["high_level"]]
    elif len(anno_data)==2:
        switch_t = sorted(list(anno_data.keys()))[1]
        if ti<switch_t:
            if ti+nt < switch_t:  # if the label is not included in the interval
                gt_high_level = 0
            else:  # the label is included in the interval
                gt_high_level = inv_d[anno_data[0]["high_level"]]
        else: # after the switching
            gt_high_level = inv_d[anno_data[switch_t]["high_level"]]
    elif len(anno_data)==3:
        switch_t = sorted(list(anno_data.keys()))[1]
        switch_t2 = sorted(list(anno_data.keys()))[2]
        if ti<switch_t:
            if ti+nt < switch_t:  # if the label is not included in the interval
                gt_high_level = 0
            else:  # the label is included in the interval
                gt_high_level = inv_d[anno_data[0]["high_level"]]
        elif ti<switch_t2: # after the switching
            if ti+nt < switch_t2:
                gt_high_level = 0
            else:
                gt_high_level = inv_d[anno_data[switch_t]["high_level"]]
        else:
            gt_high_level = inv_d[anno_data[switch_t2]["high_level"]]
    else:
        raise NotImplementedError

    return torch.tensor([gt_high_level]).float()

# get the ego state info from the nuscenes sample
def get_ego_state_from_sample(nusc, sample):
    lidar_data = nusc.get("sample_data", sample['data']["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    ego_xy = ego_pose["translation"]
    ego_euler = euler_from_quaternion(ego_pose["rotation"])
    ego_th = get_th_from_rotation(ego_pose["rotation"])
    # TODO this is hardcoded now
    ego_L = 4.0
    ego_W = 2.0
    return torch.tensor([ego_xy[0], ego_xy[1], ego_th, ego_L, ego_W]).float()


def get_nei_state_from_annotation(nusc, ann):
    other_rot = ann["rotation"]
    other_pose = ann["translation"]
    other_th = get_th_from_rotation(ann["rotation"])
    other_v = get_v_diff_for_annotation(nusc, ann)
    other_L = ann["size"][1]
    other_W = ann["size"][0]
    return torch.tensor([other_pose[0], other_pose[1], other_th, other_v, other_L, other_W]).float()

# compute the velocity profile from the annotation record
# use previous, or (if unavailable) next frame info and 
# compute the time diff  
def get_v_diff_for_annotation(nusc, ann):
    ann_token = ann["sample_token"]
    prev_token = ann['prev']
    next_token = ann['next']
    if prev_token != "":
        # check on prev
        prev = nusc.get("sample_annotation", prev_token)
        current_time = 1e-6 * nusc.get('sample', ann_token)['timestamp']
        prev_time = 1e-6 * nusc.get('sample', prev["sample_token"])['timestamp']
        time_diff = current_time - prev_time
        diff = (np.array(ann['translation']) - np.array(prev['translation'])) / time_diff
        velocity = np.linalg.norm(diff[:2])
    elif next_token != "":
        # check on future
        next0 = nusc.get("sample_annotation", next_token)
        current_time = 1e-6 * nusc.get('sample', ann_token)['timestamp']
        next_time = 1e-6 * nusc.get('sample', next0["sample_token"])['timestamp']
        time_diff = next_time - current_time
        diff = (np.array(next0['translation']) - np.array(ann['translation'])) / time_diff
        velocity = np.linalg.norm(diff[:2])
    else:
        # if both None, return zero velocity
        velocity = 0
    return velocity

# get k nearest neighbors
def get_nearest_neighbors(nusc, token, ego_state: torch.Tensor, k, ret_full=False):
    neighbors, ann_tokens = get_neighbors(nusc, token, ret_full=True)  # return a list of tensors
    nearest_neighbors = torch.zeros((k, 7))  # (valid, x, y, th, v, L, W)
    nearest_ann_tokens = []
    if len(neighbors)>0:
        neighbors = torch.stack(neighbors, dim=0)
        dist = torch.norm(ego_state.unsqueeze(0)[:, :2] - neighbors[:, :2], dim=-1)
        arg_idx = torch.argsort(dist)[:k]
        nearest_neighbors[:len(arg_idx), :1] = 1 
        nearest_neighbors[:len(arg_idx), 1:] = neighbors[arg_idx, :]
        nearest_ann_tokens = [ann_tokens[iii] for iii in arg_idx]
    if ret_full:
        return nearest_neighbors, nearest_ann_tokens
    else:
        return nearest_neighbors

# first find those annotations to locate those instances;
# then for each timestep for the sample,
# first find all the instances per step that 
# have the annotation then for the rest,
# either linear interp, or extrapolate;
def get_neighbor_trajectories(nusc, token, tokens, ego_state: torch.Tensor, k, dt, nearest_ann_tokens=None):
    if nearest_ann_tokens is None:
        _, nearest_ann_tokens = get_nearest_neighbors(nusc, token, ego_state, k, ret_full=True)
    nt = len(tokens)
    token_2_ti_d = {tok:ti for ti, tok in enumerate(tokens)}
    all_trajs = torch.zeros(k, nt, 7)
    all_trajs_idx = torch.zeros(k, nt)
    for a_i, ann_token in enumerate(nearest_ann_tokens):
        if ann_token!="":
            ann = nusc.get("sample_annotation", ann_token)
            track_list = [None for _ in range(nt)]
            cur_ann_token = ann_token
            while cur_ann_token != "":
                cur_ann = nusc.get("sample_annotation", cur_ann_token)
                nei_state = get_nei_state_from_annotation(nusc, cur_ann)
                # sync with sample
                corr_sample_token = cur_ann["sample_token"]
                if corr_sample_token not in token_2_ti_d:
                    break
                else:
                    ti = token_2_ti_d[corr_sample_token]        
                    track_list[ti] = nei_state
                cur_ann_token = cur_ann["next"]
            
            # interpolate or extrapolate the rest
            valid_idx = [i for i in range(len(track_list)) if track_list[i] is not None]
            for i in range(len(valid_idx)-1):
                begin_idx = valid_idx[i]
                end_idx = valid_idx[i+1]
                # consider all the holes in between
                for j in range(begin_idx+1, end_idx):
                    weight = (j - begin_idx) / (end_idx - begin_idx)
                    track_list[j] = track_list[begin_idx] * weight + track_list[end_idx] * (1-weight)
            all_trajs_idx[a_i, valid_idx] = 1
            # extrapolate the rest states assume constant velocity
            if valid_idx[-1] != nt - 1:
                begin_idx = valid_idx[-1]
                x, y, th, v, L, W = track_list[valid_idx[-1]]
                for j in range(begin_idx+1, nt):
                    new_x = x + v * torch.cos(th) * dt
                    new_y = y + v * torch.sin(th) * dt
                    new_v = v
                    new_th = th
                    track_list[j] = torch.tensor([new_x, new_y, new_th, new_v, L, W]).float()
                    x, y, th, v, L, W = track_list[j]

            track_list = torch.stack(track_list, dim=0)
            all_trajs[a_i, :, 1:] = track_list
            all_trajs[a_i, :, 0] = 1.0 
    return all_trajs, all_trajs_idx

def compute_t2l_dist(points, lanes, clip=False, reduced_seg=None, with_angle=False, inline=False):
    # points (N, T, 2)
    # lanes (N, T, nseg, 2)
    efficient = True

    if efficient:
        # points (N, T, 2/3)
        # lanes (N, nseg, 2/3)
        newlanes = lanes[:, None, :, :]
        n, n_segs, lane_dim = lanes.shape 
        t = points.shape[1]
        if with_angle:
            ndim = 3
        else:
            ndim = 2
        # points (n, t, 2)
        # lanes (n, 1, nseg, 2)
        point_dist = torch.norm(points[..., None, :2] - newlanes[..., :2], dim=-1)   # (n, t, nseg)
        min_idx = torch.argmin(point_dist[:, :, :-1] + point_dist[:, :, 1:], dim=2)  # (n, t)
        min_idx2 = min_idx.unsqueeze(-1).repeat(1, 1, ndim)  # (n, t, 3)
        p2 = torch.gather(lanes, dim=1, index=min_idx2)#.squeeze(1)
        p3 = torch.gather(lanes, dim=1, index=min_idx2+1)#.squeeze(1)
        x1, y1 = points[..., 0], points[..., 1]
        x2, y2 = p2[..., 0], p2[..., 1]
        x3, y3 = p3[..., 0], p3[..., 1]

        area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        bottom_l = torch.norm((p2 - p3)[..., :2], dim=-1)
        l2_dist = (torch.clamp((x1-x2)**2+(y1-y2)**2, 1e-3))**0.5

        normal_case = (bottom_l!=0).float()
        if inline:
            dist_to_lane = normal_case * area / torch.clip(bottom_l,1e-7) + (1-normal_case) * l2_dist
            l2_dist1 = (torch.clamp((x1-x3)**2+(y1-y3)**2, 1e-3))**0.5
            behind = (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2) <= 0
            ahead = (x1 - x3) * (x2 - x3) + (y1 - y3) * (y2 - y3) <= 0
            behind_all = torch.logical_and(min_idx==0, behind)
            ahead_all = torch.logical_and(min_idx==n_segs-2, ahead)
            normal = torch.logical_not(torch.logical_or(behind_all, ahead_all))
            dist = normal * dist_to_lane + behind_all * l2_dist * torch.sign(dist_to_lane) + ahead_all * l2_dist1 * torch.sign(dist_to_lane)
        else:
            dist = normal_case * area / torch.clip(bottom_l,1e-7) + (1-normal_case) * l2_dist

        if with_angle:
            lane_head = p2[..., 2]
            traj_head = points[..., 2]
            angle_dist = 1 - torch.cos(lane_head - traj_head)
            if clip:
                return torch.clip(dist.reshape(n, t), -5, 5), angle_dist.reshape(n, t)
            else:
                return dist.reshape(n, t), angle_dist.reshape(n, t)
        if clip:
            return torch.clip(dist.reshape(n, t), -5, 5)
        else:
            return dist.reshape(n, t)

    else:
        n, _, n_segs, lane_dim = lanes.shape 
        t = points.shape[1]
        assert lane_dim in [2, 3]
        if with_angle:
            points_flat = points.reshape([n*t, 1, 3])  # (n*t, 1, 3)
            lanes_flat = lanes[..., :3].reshape([n*t, n_segs, 3])  # (n*t, nseg, 3)
            point_dist = torch.norm(points_flat[..., :2] - lanes_flat[..., :2], dim=-1)
        else:
            points_flat = points.reshape([n*t, 1, 2])  # (n*t, 1, 2)
            lanes_flat = lanes[..., :2].reshape([n*t, n_segs, 2])  # (n*t, nseg, 2)
            point_dist = torch.norm(points_flat - lanes_flat, dim=-1)
        min_idx = torch.argmin(point_dist[:, :-1] + point_dist[:, 1:], dim=1)
        if with_angle:
            min_idx2 = min_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)
        else:
            min_idx2 = min_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2)
        p2 = torch.gather(lanes_flat, dim=1, index=min_idx2).squeeze(1)
        p3 = torch.gather(lanes_flat, dim=1, index=min_idx2+1).squeeze(1)
        x1, y1 = points_flat[:, 0, 0], points_flat[:, 0, 1]
        x2, y2 = p2[..., 0], p2[..., 1]
        x3, y3 = p3[..., 0], p3[..., 1]
        area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        bottom_l = torch.norm(p2 - p3, dim=-1)
        l2_dist = (torch.clamp((x1-x2)**2+(y1-y2)**2, 1e-3))**0.5
        dist = (bottom_l!=0).float() * area / torch.clip(bottom_l,1e-7) + (bottom_l==0).float() * l2_dist

        if with_angle:
            lane_head = p2[...,2] #(p2[..., 2] + p3[..., 2])/2
            traj_head = points_flat[:, 0, 2]
            angle_dist = 1 - torch.cos(lane_head - traj_head) #(bottom_l!=0).float() * area / torch.clip(bottom_l,1e-7) + (bottom_l==0).float() * l2_dist
            if clip:
                return torch.clip(dist.reshape(n, t), -5, 5), angle_dist.reshape(n, t)
            else:
                return dist.reshape(n, t), angle_dist.reshape(n, t)
        if clip:
            return torch.clip(dist.reshape(n, t), -5, 5)
        else:
            return dist.reshape(n, t)

def compute_t2l_angle(points, lanes, reduced_seg=None):
    efficient = True
    if efficient:
        newlanes = lanes[:, None, :, :]
        n, n_segs, lane_dim = lanes.shape 
        t = points.shape[1]

        point_dist = torch.norm(points[..., None, :2] - newlanes[..., :2], dim=-1)   # (n, t, nseg)
        min_idx = torch.argmin(point_dist[:, :, :-1] + point_dist[:, :, 1:], dim=2)  # (n, t)
        min_idx2 = min_idx.unsqueeze(-1).repeat(1, 1, 3)  # (n, t, 3)
        p2 = torch.gather(lanes, dim=1, index=min_idx2).squeeze(1)

        lane_head = p2[..., 2] #(p2[..., 2] + p3[..., 2])/2
        traj_head = points[..., 2]
        dist = 1 - torch.cos(lane_head - traj_head) #(bottom_l!=0).float() * area / torch.clip(bottom_l,1e-7) + (bottom_l==0).float() * l2_dist
        return dist.reshape(n, t)
    else:
        # points (N, T, 2)
        # lanes (N, T, nseg, 2)
        n, t, n_segs, lane_dim = lanes.shape 
        assert lane_dim in [2, 3]
        points_flat = points.reshape([n*t, 1, 3])  # (n*t, 1, 3)
        lanes_flat = lanes[..., :3].reshape([n*t, n_segs, 3])  # (n*t, nseg, 2)
        point_dist = torch.norm(points_flat[..., :2] - lanes_flat[..., :2], dim=-1)
        min_idx = torch.argmin(point_dist[:, :-1] + point_dist[:, 1:], dim=1)
        min_idx2 = min_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)
        p2 = torch.gather(lanes_flat, dim=1, index=min_idx2).squeeze(1)
        
        lane_head = p2[...,2]  # (p2[..., 2] + p3[..., 2])/2
        traj_head = points_flat[:, 0, 2]
        dist = 1 - torch.cos(lane_head - traj_head) #(bottom_l!=0).float() * area / torch.clip(bottom_l,1e-7) + (bottom_l==0).float() * l2_dist

        return dist.reshape(n, t)



def measure_diversity(trajs, scores, valids, nt):
    # trajs (bs, m, 3, nt * 2)
    # scores (bs, m, 3)
    # valids (bs, m, 3)
    # -------------------------
    # return ma_std_avg (, ) ma_ent_avg (, )

    # first one std loss
    trajs_np = to_np(trajs)
    acc_mask_mul_np = to_np((scores>0).long()[..., None].repeat(1, 1, 1, trajs_np.shape[-1]))
    valids_mul_np = to_np(valids[..., None].long().repeat(1, 1, 1, trajs_np.shape[-1]))
    ma_trajs_np = np.ma.masked_array(trajs_np, mask=1-acc_mask_mul_np)
    ma_trajs_std_np = np.std(ma_trajs_np, axis=1).filled(0)
    ma_trajs_std_np = np.mean(ma_trajs_std_np, axis=-1)  # feature dimension
    new_data = np.ma.masked_array(ma_trajs_std_np.flatten(), mask=(1-valids_mul_np[:, 0, :, 0]).flatten())
    ma_std_avg = np.mean(new_data)

    ma_std_avg_overall=np.mean(np.ma.masked_array(ma_trajs_std_np[:, :], mask=(1-valids_mul_np[:, 0, :, 0])), axis=-1).data
    ma_std_avg0=np.ma.masked_array(ma_trajs_std_np[:, 0], mask=(1-valids_mul_np[:, 0, 0, 0])).filled(0).data
    ma_std_avg1=np.ma.masked_array(ma_trajs_std_np[:, 1], mask=(1-valids_mul_np[:, 0, 1, 0])).filled(0).data
    ma_std_avg2=np.ma.masked_array(ma_trajs_std_np[:, 2], mask=(1-valids_mul_np[:, 0, 2, 0])).filled(0).data

    bs = trajs.shape[0]
    vol_measure_array = np.zeros((bs, 3))
    for bi in range(bs):
        for li in range(3):
            if valids[bi, 0, li] == 1:
                n_masked = np.sum(ma_trajs_np.mask[bi, :, li, 0])
                non_masked_idx = np.where(ma_trajs_np.mask[bi, :, li, 0]==0)[0]
                
                if n_masked != trajs.shape[1]:
                    # print("n_masked=",n_masked, trajs.shape[1])
                    sel_non_masked_trajs = np.array(trajs_np[bi, non_masked_idx, li].data)
                    volume = 0
                    for ti in range(nt):
                        try:
                            hull = ConvexHull(sel_non_masked_trajs[..., 2*ti:2*(ti+1)])
                            # Compute volume
                            vol= hull.volume
                        except Exception as e:
                            # print("ERROR:", e)
                            vol = 0
                        volume += vol
                    vol_measure = volume
                else:
                    vol_measure = 0
            else:
                vol_measure = 0
            vol_measure_array[bi, li] = vol_measure

    ma_vol_avg_each = np.mean(np.ma.masked_array(vol_measure_array, mask=1-to_np(valids[:, 0, :].long())))

    ma_vol_avg_overall = np.mean(np.ma.masked_array(vol_measure_array[:, :], mask=1-to_np(valids[:, 0, :].long())), axis=-1).data
    ma_vol_avg_each0 = np.ma.masked_array(vol_measure_array[:, 0], mask=1-to_np(valids[:, 0, 0].long())).filled(0).data
    ma_vol_avg_each1 = np.ma.masked_array(vol_measure_array[:, 1], mask=1-to_np(valids[:, 0, 1].long())).filled(0).data
    ma_vol_avg_each2 = np.ma.masked_array(vol_measure_array[:, 2], mask=1-to_np(valids[:, 0, 2].long())).filled(0).data

    return ma_std_avg, ma_vol_avg_each, \
            (ma_std_avg_overall, ma_std_avg0, ma_std_avg1, ma_std_avg2), (ma_vol_avg_overall, ma_vol_avg_each0, ma_vol_avg_each1, ma_vol_avg_each2)


def compute_area(x, y, th, val, bs, nt, m):
    val = val.reshape(bs*3, m, nt, 1)
    x_rel = x * torch.cos(th) + y * torch.sin(th)
    y_rel = -x * torch.sin(th) + y * torch.cos(th)
    xy_rel = torch.stack([x_rel, y_rel], dim=-1)
    xy_rel = (xy_rel * val).cpu()
    area_list=[]
    for i in range(bs * 3):
        hist, bin_edges=torch.histogramdd(xy_rel[i], bins=[100,100])
        x_length = (bin_edges[0][-1]-bin_edges[0][0])
        y_length = (bin_edges[1][-1]-bin_edges[1][0])
        area = torch.mean((hist>0).float()) * x_length * y_length
        area_list.append(area)
    return torch.mean(torch.stack(area_list))


def measure_extra_diversity(trajs, scores, valids, nt, controls, wmin, wmax, amin, amax):
    # trajs (bs, m, 3, nt * 4)
    # scores (bs, m, 3)
    # valids (bs, m, 3)

    # make them to (BS, m) or (BS, m, nt, 2) where BS=bs*3
    bs, m, _ = scores.shape
    trajs = trajs.permute(0, 2, 1, 3).reshape(bs * 3, m, nt, 4)
    scores = scores.permute(0, 2, 1).reshape(bs * 3, m)
    valids = valids.permute(0, 2, 1).reshape(bs * 3, m)

    controls = controls.permute(0, 2, 1, 3).reshape(bs * 3, m, nt, 2)

    valids = valids * ((scores>0).float())

    speed = trajs[:, :, :, 3]

    angle = (trajs[:, :, :, 2] - trajs[:, :, 0:1, 2]+np.pi)%(np.pi*2)-np.pi

    # compute scalar-wise entropy
    ent_s = compute_entropy(scores, valids)
    
    def rev(xx):
        return xx.permute(0, 2, 1).reshape(bs*3*nt, m)

    valids_rev = valids[:,None].repeat(1, nt, 1).reshape(bs*3*nt, m)

    x_ = trajs[:, :, :, 0] - trajs[:, :, 0:1, 0]
    y_ = trajs[:, :, :, 1] - trajs[:, :, 0:1, 1]
    w_ = controls[:, :, :, 0]
    a_ = controls[:, :, :, 1]

    ent_w = compute_entropy(rev(w_), valids_rev, x_min=wmin, x_max=wmax)
    ent_a = compute_entropy(rev(a_), valids_rev, x_min=amin, x_max=amax)
    area_xy = compute_area(x_, y_, trajs[:, :, :, 2], valids_rev, bs, nt, m)
    results = {
        "ent_s": torch.mean(ent_s),
        "ent_w": torch.mean(ent_w),
        "ent_a": torch.mean(ent_a),
        "ent_wa": torch.mean(ent_w)+torch.mean(ent_a),
        "area": area_xy,
    }
    return results

def main():
    num_trials = 5
    old_t_list=[]
    new_t_list=[]
    dataroot=os.environ["MY_DATA_DIR"]
    if len(dataroot)<1:
        exit("CANNOT FIND ENV VARIABLE:%s"%(dataroot))
    else:
        dataroot=os.path.join(dataroot, "nuscenes")
    for i in range(num_trials+1):
        t1=time.time()
        nusc1 = NuscenesPkl(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        t2=time.time()
        nusc2 = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        t3=time.time()
        new_t_list.append(t2-t1)
        old_t_list.append(t3-t2)
        del nusc1
        del nusc2

    print(" ".join(["%.4f"%x for x in old_t_list]))
    print(" ".join(["%.4f"%x for x in new_t_list]))

    import numpy as np
    old_t_avg = np.mean(old_t_list[1:])
    new_t_avg = np.mean(new_t_list[1:])

    print("old:%.4f  new:%.4f"%(old_t_avg, new_t_avg))


if __name__ == "__main__":
    main()