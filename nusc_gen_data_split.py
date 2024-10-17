import os
import time
import nusc_api as napi
import torch

def generate_data_split(cache_d, seed, train_ratio, nt, mini, mixed):
    torch.manual_seed(seed)
    filter_list = [181, 391, 406, 55, 108, 394, 38, 45, 492, 265] + [569, 79] + [304, 506, 570, 571, 594] #(safe dist violation)
    if mini in cache_d:
        nusc, nusc_map_d, meta_list = cache_d[mini]
    else:
        nusc, nusc_map_d = napi.get_nuscenes(is_mini=mini)
        meta_list = napi.get_scene_tokens(nusc)
        cache_d[mini] = nusc, nusc_map_d, meta_list

    indices = []
    indices_d = {"train":[], "val":[]}
    if mixed:       
        for traj_i, tokens in meta_list:
            if traj_i in filter_list:
                continue
            for ti in range(1, len(tokens) - nt + 1):
                indices.append((traj_i, ti, tokens[ti]))
        rridx = torch.randperm(len(indices))
        new_train_len = int(len(indices) * train_ratio)
        indices_d["train"] = sorted([indices[idxx] for idxx in rridx[:new_train_len]], key=lambda x:x[0] * 100000 + x[1])
        indices_d["val"] = sorted([indices[idxx] for idxx in rridx[new_train_len:]], key=lambda x:x[0] * 100000 + x[1])
    else:
        train_len = int(len(meta_list) * train_ratio)
        for traj_i, tokens in meta_list[:train_len]:
            if traj_i in filter_list:
                continue
            for ti in range(1, len(tokens) - nt + 1):
                indices_d["train"].append((traj_i, ti, tokens[ti]))
        for traj_i, tokens in meta_list[train_len:]:
            if traj_i in filter_list:
                continue
            for ti in range(1, len(tokens) - nt + 1):
                indices_d["val"].append((traj_i, ti, tokens[ti]))

    for split in ["train", "val"]:
        with open("data/%s%s%s_split.txt"%("mini_" if mini else "", "mixed_" if mixed else "", split), "w") as f:
            for res in indices_d[split]:
                traj_i, ti, tokens_i = res
                line="%s %s %s"%(traj_i, ti, tokens_i) 
                f.write(line+"\n")


def main():
    os.makedirs("data", exist_ok=True)
    seed = 1007
    nt = 20
    cache_d = {}
    train_ratio = 0.7
    generate_data_split(cache_d, seed, train_ratio, nt, mini=True, mixed=True)
    generate_data_split(cache_d, seed, train_ratio, nt, mini=True, mixed=False)
    generate_data_split(cache_d, seed, train_ratio, nt, mini=False, mixed=True)
    generate_data_split(cache_d, seed, train_ratio, nt, mini=False, mixed=False)


if __name__ == "__main__":
    t1=time.time()
    main()
    t2=time.time()