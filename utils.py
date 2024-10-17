import os
from os.path import join as ospj
import sys
import time
import math
import shutil
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import imageio
import matplotlib.pyplot as plt

def get_data_dir():
    dataroot=os.environ["MY_DATA_DIR"]
    if len(dataroot)<1:
        exit("CANNOT FIND ENV VARIABLE for 'MY_DATA_DIR':%s"%(dataroot))
    else:
        dataroot=os.path.join(dataroot, "nuscenes")
    return dataroot

def get_exp_dir():
    return "exps/"

def get_model_path(pretrained_path):
    return ospj(get_exp_dir(), smart_path(pretrained_path))

def find_path(path):
    return os.path.join(get_exp_dir(), path)

def find_npz_path(path):
    if ".npz" not in path:
        path = os.path.join(path, "cache.npz")
    if path.startswith("/"):
        path = path
    else:
        path = os.path.join(get_exp_dir(), path)
    return path

def smart_path(s):
    if ".ckpt" not in s:
        s = s+"/models/model_last.ckpt"
    return s

def to_np(x):
    return x.detach().cpu().numpy()

def to_torch(x):
    return torch.from_numpy(x).float().cuda()

def uniform_tensor(amin, amax, size):
    return torch.rand(size) * (amax - amin) + amin

def rand_choice_tensor(choices, size):
    return torch.from_numpy(np.random.choice(choices, size)).float()

def to_np_dict(di):
    di_np = {}
    for key in di:
        di_np[key] = to_np(di[key])
    return di_np

def dict_to_cuda(batch):
    cuda_batch = {}
    for key in batch:
        cuda_batch[key] = batch[key]
        if hasattr(batch[key], "device"):
            cuda_batch[key] = cuda_batch[key].cuda()
    return cuda_batch

def dict_to_torch(batch, keep_keys=[]):
    torch_batch = {}
    for key in batch:
        if key in keep_keys:
            torch_batch[key] = batch[key]
        else:
            torch_batch[key] = torch.from_numpy(batch[key])
    return torch_batch

def save_model_freq_last(state_dict, model_dir, epi, save_freq, epochs):
    if epi % save_freq == 0 or epi == epochs-1:
        torch.save(state_dict, "%s/model_%05d.ckpt"%(model_dir, epi))
    if epi % 10 == 0 or epi == epochs-1:
        torch.save(state_dict, "%s/model_last.ckpt"%(model_dir))

def plt_save_close(img_path, bbox_inches='tight', pad_inches=0.1):
    plt.savefig(img_path, bbox_inches=bbox_inches, pad_inches=pad_inches)
    plt.close()

def build_relu_nn(input_dim, output_dim, hiddens, activation_fn=torch.nn.ReLU, last_fn=None):
    n_neurons = [input_dim] + hiddens + [output_dim]
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        layers.append(activation_fn())
    if last_fn is not None:
        layers[-1] = last_fn()
    else:
        del layers[-1]
    return nn.Sequential(*layers)

def build_relu_nn1(input_output_dim, hiddens, activation_fn, last_fn=None):
    return build_relu_nn(input_output_dim[0], input_output_dim[1], hiddens, activation_fn, last_fn=last_fn)

def generate_gif(gif_path, duration, fs_list):
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for filename in fs_list:
            image = imageio.imread(filename)
            writer.append_data(image)

class MyTimer():
    def __init__(self):
        self.timestamp = {}
        self.count = {}
        self.profile = {}
        self.left = {}
        self.right = {}
        self.last = None
    
    def add(self, key, new_name=None):
        self.timestamp[key] = time.time()
        if key not in self.count:
            self.count[key] = 0 
        self.count[key] += 1
        
        if self.last is not None and self.count[key]==self.count[self.last]:
            if new_name is None:
                new_name = "%s-%s"%(key, self.last)
            self.left[new_name] = key
            self.right[new_name] = self.last
            dt = self.timestamp[key] - self.timestamp[self.last]
            if new_name not in self.profile:
                self.profile[new_name] = 0
            self.profile[new_name] += dt
        
        self.last = key
    
    def print_profile(self):
        s=""
        for key in self.profile:
            left = self.left[key]
            right = self.right[key]
            tsum = self.profile[key]
            cnt = self.count[left]
            s += "%s:%.3f "%(key, tsum/ cnt)
        print(s)


class EtaEstimator():
    def __init__(self, start_iter, end_iter, check_freq=1, epochs=None, total_train_bs=None, total_val_bs=None, batch_size=None, viz_freq=None, num_workers=1):
        self.start_iter = start_iter
        num_workers = 1# if num_workers is None else num_workers
        self.end_iter = end_iter//num_workers
        self.check_freq = check_freq
        self.curr_iter = start_iter
        self.start_timer = None
        self.interval = 0
        self.eta_t = 0
        self.num_workers = num_workers

        self.viz_freq = viz_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.prev_is_viz = False
        self.prev_timer = time.time()
        self.nn_stat_train = []
        self.nn_train_bs = []
        self.nn_stat_val = []
        self.nn_val_bs = []
        self.viz_stat = []
        self.total_train_bs = total_train_bs
        self.total_val_bs = total_val_bs

    def update(self):
        if self.start_timer is None:
            self.start_timer = time.time()
        self.curr_iter += 1
        # if self.curr_iter % (max(1,self.check_freq//self.num_workers)) == 0:
        self.interval = self.elapsed() / (self.curr_iter - self.start_iter)        
        self.eta_t = self.interval * (self.end_iter - self.curr_iter)
    
    def update_viz_time(self, duration):
        self.viz_stat.append(duration)

    def smart_update(self, epi, duration=None, bs=None, mode=None, bi=None, is_viz=False):
        self.curr_epi = epi
        self.curr_stage = mode
        self.curr_bi = bi
        nn_time = duration
        if mode=="train":
            self.nn_stat_train.append(nn_time)
            self.nn_train_bs.append(bs)
        elif mode=="val":
            self.nn_stat_val.append(nn_time)
            self.nn_val_bs.append(bs)
        else:
            raise NotImplementedError
        
        if len(self.nn_stat_train)>0:
            if len(self.nn_stat_train)>1:
                nn_train_sum = np.sum(self.nn_stat_train[1:])
                nn_train_bs = np.sum(self.nn_train_bs[1:])
            else:
                nn_train_sum = np.sum(self.nn_stat_train)
                nn_train_bs = np.sum(self.nn_train_bs)
            nn_train_avg_per_sample = nn_train_sum / nn_train_bs
        if len(self.nn_stat_val)>0:
            nn_val_sum = np.sum(self.nn_stat_val)
            nn_val_bs = np.sum(self.nn_val_bs)
            nn_val_avg_per_sample = nn_val_sum / nn_val_bs
        else:
            nn_val_avg_per_sample = nn_train_avg_per_sample
        remain_epis = self.epochs - self.curr_epi - 1
        if self.curr_stage == "train":
            remain_train_bs = self.total_train_bs - (self.curr_bi+1) * self.batch_size
            remain_val_bs = self.total_val_bs
             
        elif self.curr_stage == "val":
            remain_train_bs = 0
            remain_val_bs = self.total_val_bs - (self.curr_bi+1) * self.batch_size

        remain_single_time = remain_train_bs * nn_train_avg_per_sample+ remain_val_bs * nn_val_avg_per_sample
       
        if len(self.viz_stat)>0:
            avg_viz_time = np.mean(self.viz_stat)
        else:
            avg_viz_time = 1 * nn_train_avg_per_sample

        viz_cnt=0
        for ii in range(self.curr_epi, self.epochs):
            if ii % self.viz_freq == 0 or ii == self.epochs-1:
                viz_cnt += 1
        remain_viz_time = viz_cnt * avg_viz_time
        self.eta_t_smart = remain_epis * (nn_train_avg_per_sample*self.total_train_bs + nn_val_avg_per_sample*self.total_val_bs) +\
                           remain_single_time + remain_viz_time
        
        if is_viz==False:
            self.update()
                
        self.prev_is_viz = is_viz
        self.prev_timer = time.time()

    def elapsed(self):
        return time.time() - self.start_timer
    
    def eta(self):
        return self.eta_t
    
    def elapsed_str(self):
        return time_format(self.elapsed())
    
    def interval_str(self):
        return time_format(self.interval)

    def eta_str(self):
        return time_format(self.eta_t)

    def eta_str_smart(self):
        return time_format(self.eta_t_smart)

def time_format(secs):
    _s = secs % 60 
    _m = secs % 3600 // 60
    _h = secs % 86400 // 3600
    _d = secs // 86400
    if _d != 0:
        return "%02dD%02dh%02dm%02ds"%(_d, _h, _m, _s)
    else:
        if _h != 0:
            return "%02dH%02dm%02ds"%(_h, _m, _s)
        else:
            if _m != 0:
                return "%02dm%02ds"%(_m, _s)
            else:
                return "%05.2fs"%(_s)


def uniform(a, b, size):
    return torch.rand(*size) * (b - a) + a

def linspace(a, b, size):
    return torch.from_numpy(np.linspace(a, b, size)).float()

# TODO logger
class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def write_cmd_to_file(log_dir, argv):
    with open(ospj(log_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(argv))

# TODO create the exp directory
def setup_exp_and_logger(args, set_gpus=True, test=False):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.stdout = logger = Logger()
    EXP_ROOT_DIR = get_exp_dir()
    if test:
        if hasattr(args, "rl") and args.rl:
            tuples = args.rl_path.split("/")
        else:
            tuples = args.net_pretrained_path.split("/")
        if ".ckpt" in tuples[-1] or ".zip" in tuples[-1] :
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-3])
        else:
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-1])
        if hasattr(args, "suffix") and args.suffix is not None:
            suffix="_"+args.suffix
        else:
            suffix=""
        args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "test_%s%s" % (logger._timestr, suffix))
    else:
        if args.exp_name.startswith("e"):
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, args.exp_name)
        else:
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_%s" % (logger._timestr, args.exp_name))
    args.viz_dir = os.path.join(args.exp_dir_full, "viz")
    args.src_dir = os.path.join(args.exp_dir_full, "src")
    args.model_dir = os.path.join(args.exp_dir_full, "models")
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(args.src_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    for fname in os.listdir('./'):
        if fname.endswith('.py'):
            shutil.copy(fname, os.path.join(args.src_dir, fname))

    logger.create_log(args.exp_dir_full)
    write_cmd_to_file(args.exp_dir_full, sys.argv)
    np.savez(os.path.join(args.exp_dir_full, 'args'), args=args)

    if set_gpus and hasattr(args, "gpus") and args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    return args

# TODO metrics
class MeterDict:
    def __init__(self):
        self.d = {}
    
    def reset(self):
        del self.d
        self.d = {}

    def update(self, key, val):
        if key not in self.d:
            # curr, count, avg
            self.d[key] = [val, 1, val]
        else:
            _, count, avg = self.d[key]
            self.d[key][0] = val
            self.d[key][1] = count+1
            ratio = 1 / (count+1)
            self.d[key][2] = avg * (1-ratio) + val * ratio
    
    def get_val(self, key):
        return self.d[key][0]

    def __getitem__(self, key):
        return self.get_val(key)

    def get_avg(self, key):
        return self.d[key][2]
    
    def __contains__(self, key):
        return key in self.d

    def __call__(self, key):
        return self.get_avg(key)


def compute_entropy(x, mask, n_bins=10, x_min=None, x_max=None):
    # x (N, m)
    # mask (N, m)
    # return (N, )
    assert len(x.shape)==len(mask.shape)==2
    BIG_NUM = float("Inf")
    SMALL_NUM = float("-Inf")
    CLIP_VAL=1e-5
    x_aug_min = x * 1.0
    x_aug_min[mask==0] = SMALL_NUM
    x_aug_max = x * 1.0
    x_aug_max[mask==0] = BIG_NUM
    
    if x_min is None:
        xmin = torch.min(x_aug_max, dim=1)[0] - CLIP_VAL
        xmax = torch.max(x_aug_min, dim=1)[0] + CLIP_VAL
    else:
        xmin = x_min * torch.ones_like(x[:,0])
        xmax = x_max * torch.ones_like(x[:,0])

    # gap = (xmax - xmin) / n_bins
    alphas = torch.linspace(0.0, 1.0, n_bins+1)[None, :].to(x.device)
    bins = xmin[:, None] * (1 - alphas) + xmax[:, None] * alphas  # (N, 11)
    
    # probs = torch.floor((x - xmin) / gap)
    spotted = torch.logical_and(x_aug_max[:, :, None]>=bins[:, None, :-1], x_aug_max[:, :, None]<bins[:, None, 1:])  # (N, m, 10)
    counts = torch.sum(spotted.float(), dim=1)  # (N, m)
    probs = counts / torch.clip(torch.sum(counts, dim=-1, keepdim=True), CLIP_VAL)
    entropy = torch.sum(-probs * torch.log2(torch.clip(probs, CLIP_VAL)), dim=-1)
    return entropy


def euler_from_quaternion(quat):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def generate_bbox(x, y, theta, L, W):
    # (2, 5)
    bbox=np.array([
        [L/2, W/2],
        [L/2, -W/2],
        [-L/2, -W/2],
        [-L/2, W/2],
    ]).T

    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

    # (2, 1)
    trans = np.array([[
        x, y
    ]]).T
    new_bbox = (rot @ bbox) + trans
    return new_bbox.T


def get_anchor_point(x, y, th, L, W, num_L, num_W):
    x1 = L/2
    y1 = W/2
    x2 = -L/2
    y2 = W/2
    x3 = -x1
    y3 = -y1
    x4 = -x2
    y4 = -y2
    r_l = L / num_L / 2
    r_w = W / num_W / 2
    r = torch.minimum(torch.maximum(r_l, r_w), W / 2)

    poly = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1).reshape(list(x1.shape) + [4, 2])
    poly_x = poly[..., 0] * torch.cos(th[..., None]) - poly[..., 1] * torch.sin(th[..., None]) + x[..., None]
    poly_y = poly[..., 0] * torch.sin(th[..., None]) + poly[..., 1] * torch.cos(th[..., None]) + y[..., None]
    poly = torch.stack([poly_x, poly_y], dim=-1)

    alpha = torch.linspace(0, 1, num_L).to(x1.device)
    beta = torch.linspace(0, 1, num_W).to(x1.device)
    xs_ = (x2 + r)[..., None] * (1 - alpha) + (x1 - r)[..., None] * alpha # (N, T, k1)
    ys_ = (y3 + r)[..., None] * (1 - beta) + (y2 - r)[..., None] * beta # (N, T, k2)
    # xys_ = torch.stack(torch.meshgrid(xs_, ys_), dim=-1).reshape(list(xs.shape[:-1]) + [num_L*num_W, 2])

    batch_size = list(x1.shape)
    xs_ = xs_[..., None].expand(batch_size+ [num_L, num_W]).reshape(batch_size +[num_L*num_W])
    ys_ = ys_[..., None, :].expand(batch_size+ [num_L, num_W]).reshape(batch_size +[num_L*num_W])
    # print(xs_.shape, ys_.shape, th.shape, th[..., None].shape)
    xs = xs_ * torch.cos(th[..., None]) - ys_ * torch.sin(th[..., None]) + x[..., None]
    ys = xs_ * torch.sin(th[..., None]) + ys_ * torch.cos(th[..., None]) + y[..., None]
    # xys = torch.stack(torch.meshgrid(xs, ys), dim=-1).reshape(list(xs.shape[:-1]) + [num_L*num_W, 2])  # (N, T, k1*k2, 2)
    xys = torch.stack([xs, ys], dim=-1)
    return poly, xys, r

def dist_between_two_cars(x1, y1, th1, L1, W1, x2, y2, th2, L2, W2, num_L, num_W, debug=False, full=False):
    poly1, xys1, rs1 = get_anchor_point(x1, y1, th1, L1, W1, num_L, num_W)  # (K, 1, n, k, 2), (K, 1, n, )
    poly2, xys2, rs2 = get_anchor_point(x2, y2, th2, L2, W2, num_L, num_W)  # (K, m, 1, k, 2), (K, m, 1, )
    dist = torch.norm(xys1[..., None, :] - xys2[..., None, :, :], dim=-1)
    dist = dist.reshape(list(dist.shape[:-2]) + [num_L * num_W * num_L * num_W])
    # (K, 1, n, k, 1, 2) - (K, m, 1, 1, k, 2) -> (K, m, n, k, k) -> (K, m, n)
    min_dist = torch.min(dist, dim=-1)[0]
    car_dist = min_dist - rs1 - rs2
    if full:
        return car_dist, min_dist, rs1 + rs2
    else:
        return car_dist

def dist_between_two_cars_stack(state1, state2, num_L, num_W, debug=False, ego_L=None, ego_W=None, full=False):
    if ego_L is not None:
        assert 6>= state2.shape[-1] >= 5
        return dist_between_two_cars(
                state1[..., 0], state1[..., 1], state1[..., 2], ego_L * torch.ones_like(state1[..., 0]), ego_W * torch.ones_like(state1[..., 0]),  
                state2[..., 0], state2[..., 1], state2[..., 2], state2[..., -2], state2[..., -1],
            num_L, num_W, debug, full)
    else:
        # print(state1.shape, state2.shape)
        assert 6>= state1.shape[-1] >= 5
        assert 6>= state2.shape[-1] >= 5
        return dist_between_two_cars(
                state1[..., 0], state1[..., 1], state1[..., 2], state1[..., -2], state1[..., -1],  
                state2[..., 0], state2[..., 1], state2[..., 2], state2[..., -2], state2[..., -1],
            num_L, num_W, debug, full)