import numpy as np
import torch
import torch.nn as nn
from stl_d_lib import *
import utils


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.input_dim = 77  # hybrid for both multi-lane and intersection
        self.output_dim = args.nt * 2
        self.feat_dim = feat_dim = 32
        self.stlp_dim = stlp_dim = 6  # (vmin, vmax, dmin, dmax, d_safe)
        self.lane_dim = 3
        self.n_segs = args.n_segs
        self.time_dim = 32

        self.ego_encoder = utils.build_relu_nn(6, feat_dim, args.hiddens, activation_fn=nn.ReLU)
        self.neighbor_encoder = utils.build_relu_nn(7, feat_dim, args.hiddens, activation_fn=nn.ReLU)
        self.lane_encoder = utils.build_relu_nn(self.n_segs * self.lane_dim, feat_dim, args.hiddens, activation_fn=nn.ReLU)

        if self.args.diffusion:
            latent_dim = args.nt * 2 + self.time_dim + 1 + stlp_dim  # (noise + timestep + high_level decision + stlp)
        elif self.args.bc:
            latent_dim = 1 + stlp_dim
        elif self.args.vae:
            latent_dim = args.vae_dim + 1 + stlp_dim # (noise + high_level decision + stlp)
            self.traj_encoder = utils.build_relu_nn(args.nt * 2, args.vae_dim * 2, args.hiddens, activation_fn=nn.ReLU)
        else:
            latent_dim = 1 + stlp_dim  # (stl + highlevel)            
        
        if args.use_init_hint:
            latent_dim += args.nt * 2

        self.policy_net = utils.build_relu_nn(latent_dim + feat_dim * 7, args.nt * 2, args.hiddens, activation_fn=nn.ReLU)
        if self.args.rect_head:
            rect_out_dim = args.nt * 2
            extra_in_dim = 0
            if self.args.diverse_loss and self.args.no_arch==False and self.args.diverse_fuse_type=="cat":
                extra_in_dim += args.nt * 2

            if self.args.diverse_loss:
                self.merge_net = utils.build_relu_nn(args.nt*2, args.nt*2, [32, 32], activation_fn=nn.ReLU)
            self.rect_net = utils.build_relu_nn(latent_dim - self.time_dim + feat_dim * 7 + extra_in_dim, rect_out_dim, args.rect_hiddens, activation_fn=nn.ReLU)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def encode_feat(self, nn_input, ext=None):
        bs = nn_input["ego_traj"].shape[0]

        # normalization
        ego = nn_input["ego_traj"][:, 0]
        ego_un = ego.unsqueeze(1)

        # neighbor normalization
        neis_ = nn_input["neighbors"]
        neis_xyth = normalize_xyth(neis_[..., 1:4], ego_un, neis_[..., 0])
        neis_input = torch.cat([neis_[..., 0:1], neis_xyth, neis_[..., 4:7]], dim=-1)
        
        # lane normalization
        tmp_di = {}
        for key in ["curr", "left", "right"]:
            item = normalize_xyth(nn_input["%slane_wpts"%(key)], ego_un, nn_input["%s_id"%(key)])
            tmp_di[key] = item
        lanes = torch.stack((tmp_di["curr"], tmp_di["left"], tmp_di["right"]), dim=1)  # (N, 3, nseg, 3)
        lanes_start = lanes[..., 0:1, :]  # use difference encoding 
        lanes_diff = lanes[..., 1:, :] - lanes[..., :-1, :]
        segs = lanes.shape[-2]
        lanes_input = torch.cat([lanes_start, lanes_diff], dim=-2).reshape(bs, 3, segs * self.lane_dim)
        
        ego_xyth = normalize_xyth(ego[..., :3], ego[..., :3])
        ego_input = torch.cat([ego_xyth, ego[..., 3:]], dim=-1)

        # encoder part
        ego_feat = self.ego_encoder(ego_input)  # (N, nfeat)

        nei_feat = self.neighbor_encoder(neis_input)  # (N, n_neis, nfeat)
        nei_feat_min = torch.min(nei_feat, dim=1)[0]
        nei_feat_avg = torch.mean(nei_feat, dim=1)
        nei_feat_max = torch.max(nei_feat, dim=1)[0]
        nei_feat = torch.cat([nei_feat_min, nei_feat_avg, nei_feat_max], dim=-1)  # (N, 2 * nfeat)
        # nei_feat = nei_feat_max

        lanes_feat = self.lane_encoder(lanes_input) # (N, 3, nfeat)
        lanes_feat = lanes_feat.reshape(bs, -1)
        feature = torch.cat([ego_feat, nei_feat, lanes_feat], dim=-1)  # (N, 5 * nfeat)

        return feature

    def forward(self, nn_input, ext=None, get_feature=False, prev_feature=None, sample=False, n_randoms=None):
        bs = nn_input["ego_traj"].shape[0]
        multi_check = any([self.args.diffusion, self.args.vae, self.args.bc]) and self.args.gt_data_training==False

        if prev_feature is not None:
            feature = prev_feature
        else:
            feature = self.encode_feat(nn_input)
            if multi_check:
                k = feature.shape[-1]
                if n_randoms is None:
                    n_randoms = self.args.n_randoms
                n_rep = n_randoms * 3
                feature = feature.reshape(bs, 1, k).repeat(1, n_rep, 1).reshape(-1, k)

        n = feature.shape[0]  # n might be not bs
        if multi_check:
            stlp_dense_feat = nn_input["stlp_dense"][:,0]
        else:
            stlp_dense_feat = ext["gt_stlp"]

        if self.args.diffusion:
            time_feat = self.pos_encoding(ext["timestep"], self.time_dim)
            if multi_check:
                policy_input = torch.cat([feature, ext["noise"], time_feat, ext["highlevel"], stlp_dense_feat], dim=-1)  # (n, args.nt * 2)
            else:
                n_rep = self.args.n_randoms
                feature_tmp = feature.reshape(bs, 1, -1).repeat(1, n_rep, 1).reshape(bs * n_rep, -1)
                highlevel_tmp = ext["highlevel"].reshape(bs, 1, -1).repeat(1, n_rep, 1).reshape(bs * n_rep, -1)
                stlp_tmp = stlp_dense_feat.reshape(bs, 1, -1).repeat(1, n_rep, 1).reshape(bs * n_rep, -1)
                policy_input = torch.cat([feature_tmp, ext["noise"], time_feat, highlevel_tmp, stlp_tmp], dim=-1)  # (n, args.nt * 2)
        elif self.args.bc:
            policy_input = torch.cat([feature, ext["highlevel"], stlp_dense_feat], dim=-1)
        elif self.args.vae:
            if sample is not False:
                latent = sample
                latent_mean, latent_logstd, latent_std = None, None, None
            else:
                if multi_check:
                    code = self.traj_encoder(ext["trajopt_controls"].reshape(-1, self.args.nt * 2))
                else:
                    n_rep = self.args.n_randoms
                    feature_tmp = feature.reshape(bs, 1, -1).repeat(1, n_rep, 1).reshape(bs * n_rep, -1)
                    highlevel_tmp = ext["highlevel"].reshape(bs, 1, -1).repeat(1, n_rep, 1).reshape(bs * n_rep, -1)
                    stlp_tmp = stlp_dense_feat.reshape(bs, 1, -1).repeat(1, n_rep, 1).reshape(bs * n_rep, -1)
                    code = self.traj_encoder(ext["gt_controls"].reshape(-1, self.args.nt * 2))
                    code = code[:,None,:].repeat(1, self.args.n_randoms, 1).reshape(bs * n_rep, self.args.vae_dim*2)
                latent_mean = code[..., :self.args.vae_dim]
                latent_logstd = code[..., self.args.vae_dim:]
                latent_std = torch.exp(latent_logstd)
                latent = ext["noise"] * latent_std + latent_mean
                
            if multi_check:
                policy_input = torch.cat([feature, latent, ext["highlevel"], stlp_dense_feat], dim=-1)  # (n, args.nt * 2)
            else:
                policy_input = torch.cat([feature_tmp, latent, highlevel_tmp, stlp_tmp], dim=-1)  # (n, args.nt * 2)
        else:
            policy_input = torch.cat([feature, nn_input["gt_high_level"], stlp_dense_feat], dim=-1) 
        
        if self.args.use_init_hint:
            policy_input = torch.cat([policy_input, nn_input["params_init"].reshape(list(policy_input.shape[:-1]) + [self.args.nt * 2])], dim=-1)

        raw_controls = self.policy_net(policy_input)

        if self.args.diffusion:
            raw_controls = raw_controls + ext["noise"]
        
        raw_controls = raw_controls.reshape(-1, self.args.nt, 2)

        if self.args.diffusion:
            steer = raw_controls[..., 0] # * self.args.mul_w_max
            accel = raw_controls[..., 1] # * self.args.mul_a_max
        else:
            steer = torch.nn.Tanh()(raw_controls[..., 0]) * self.args.mul_w_max
            accel = torch.nn.Tanh()(raw_controls[..., 1]) * self.args.mul_a_max
        controls = torch.stack([steer, accel], dim=-1)

        if get_feature:
            return controls, feature
        else:
            if self.args.vae:
                return controls, latent_mean, latent_logstd, latent_std
            else:
                return controls
    
    def rect_forward(self, feature, highlevel, stlp_dense_feat, init_controls, scores, extras=None):
        n = feature.shape[0]
        # print(feature.shape, highlevel.shape, stlp_dense_feat.shape, init_controls.shape)
        if self.args.diverse_loss and self.args.no_arch==False:
            fused_controls = self.merge_net(init_controls.reshape(-1, self.args.nt*2))
            bs =int(init_controls.shape[0] / 3 / self.args.n_randoms)
            fused_controls = fused_controls.reshape(bs, self.args.n_randoms, 3, self.args.nt*2)
            fused_controls = fused_controls.permute(0, 2, 1, 3)
            
            NS = self.args.n_shards

            fused_controls = fused_controls.reshape(bs, 3, NS, self.args.n_randoms // NS, self.args.nt*2)
            fused_controls = torch.max(fused_controls, dim=3, keepdim=True)[0]
            # print("Fused",fused_controls.shape)
            fused_controls = fused_controls.repeat(1, 1, 1, self.args.n_randoms // NS, 1).reshape(bs, 3, self.args.n_randoms, self.args.nt*2)       
            fused_controls = fused_controls.permute(0, 2, 1, 3)
            fused_controls = fused_controls.reshape(init_controls.shape[0], self.args.nt, 2)
            if self.args.diverse_fuse_type=="add":
                fused_controls = init_controls + fused_controls
            if self.args.diverse_fuse_type=="cat":
                policy_input = torch.cat([feature, highlevel, stlp_dense_feat, init_controls.reshape(n, self.args.nt*2), fused_controls.reshape(n, self.args.nt*2)], dim=-1)
            elif self.args.diverse_fuse_type=="add":
                policy_input = torch.cat([feature, highlevel, stlp_dense_feat, fused_controls.reshape(n, self.args.nt*2)], dim=-1)
            else:
                raise NotImplementedError
        else:
            policy_input = torch.cat([feature, highlevel, stlp_dense_feat, init_controls.reshape(n, self.args.nt*2)], dim=-1)
        raw_controls_aug = self.rect_net(policy_input)
        raw_controls_aug = raw_controls_aug.reshape(n, self.args.nt, 2)
        
        if self.args.interval:
            init_w = init_controls[..., 0]
            init_a = init_controls[..., 1]
            raw_controls = torch.nn.Tanh()(raw_controls_aug)
            w_mask = (raw_controls[..., 0]>=0).float()
            a_mask = (raw_controls[..., 1]>=0).float()
            raw_controls_w0 = raw_controls[..., 0] * (init_w-(-self.args.mul_w_max))
            raw_controls_w1 = raw_controls[..., 0] * (self.args.mul_w_max-init_w)
            raw_controls_a0 = raw_controls[..., 1] * (init_a-(-self.args.mul_a_max))
            raw_controls_a1 = raw_controls[..., 1] * (self.args.mul_a_max-init_a)
            w_merge = raw_controls_w0 * (1-w_mask) + raw_controls_w1 * w_mask
            a_merge = raw_controls_a0 * (1-a_mask) + raw_controls_a1 * a_mask
            raw_controls = torch.stack([w_merge, a_merge], dim=-1)
        else:
            raw_controls = raw_controls_aug

        violated=((scores<0).float()[:,None,None])
        raw_controls = init_controls + raw_controls * violated
        if self.args.clip_rect:
            w_merge = torch.clip(raw_controls[..., 0], -self.args.mul_w_max, self.args.mul_w_max)
            a_merge = torch.clip(raw_controls[..., 1], -self.args.mul_a_max, self.args.mul_a_max)
            raw_controls = torch.stack([w_merge, a_merge], dim=-1)

        return raw_controls


def normalize_xyth(state, base, valid=None, no_theta=False):
    assert len(state.shape) == len(base.shape) and state.shape[0]==base.shape[0]
    x = state[..., 0]
    y = state[..., 1]
    if no_theta==False:
        th = state[..., 2]
    base_x = base[..., 0]
    base_y = base[..., 1]
    base_th = base[..., 2]
    if valid is not None:
        x_trans = x - base_x * valid
        y_trans = y - base_y * valid
    else:
        x_trans = x - base_x
        y_trans = y - base_y
    x_rel = x_trans * torch.cos(base_th) + y_trans * torch.sin(base_th)
    y_rel = -x_trans * torch.sin(base_th) + y_trans * torch.cos(base_th)
    
    if no_theta==False:
        if valid is not None:
            th_rel = th - base_th * valid
        else:
            th_rel = th - base_th
        return torch.stack([x_rel, y_rel, th_rel], dim=-1)
    else:
        return torch.stack([x_rel, y_rel], dim=-1)

def normalize_xyth_np(state, base, valid=None, no_theta=False):
    assert len(state.shape) == len(base.shape) and state.shape[0]==base.shape[0]
    x = state[..., 0]
    y = state[..., 1]
    if no_theta==False:
        th = state[..., 2]
    base_x = base[..., 0]
    base_y = base[..., 1]
    base_th = base[..., 2]
    if valid is not None:
        x_trans = x - base_x * valid
        y_trans = y - base_y * valid
    else:
        x_trans = x - base_x
        y_trans = y - base_y
    x_rel = x_trans * np.cos(base_th) + y_trans * np.sin(base_th)
    y_rel = -x_trans * np.sin(base_th) + y_trans * np.cos(base_th)
    if no_theta==False:
        if valid is not None:
            th_rel = th - base_th * valid
        else:
            th_rel = th - base_th
        return np.stack([x_rel, y_rel, th_rel], axis=-1)
    else:
        return np.stack([x_rel, y_rel], axis=-1)