def PredFormer_2D(in_shape, pred_n_frames_per_step):
    config = { "model_config": {
        # model
        # image h w c
        'height': in_shape[-2],
        'width': in_shape[-1],
        'num_channels': in_shape[1],
        # video length in and out
        'pre_seq': in_shape[0],
        'after_seq': pred_n_frames_per_step,
        # patch size
        'patch_size': 8,
        'dim': 256, 
        'heads': 8,
        'dim_head': 32,
        # dropout
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'drop_path': 0.25,
        'scale_dim': 2,
        # depth
        'depth': 1,
        'Ndepth': 2, # For FullAttention-8, for BinaryST, BinaryST, FacST, FacTS-4, for TST,STS-3, for TSST, STTS-2
    }}
    return config

def SimVP_2D(in_shape, pred_n_frames_per_step):
    config = {
        # model
        "in_shape": in_shape,
        "hid_S": 16,
        "hid_T": 256,
        "N_S": 4,
        "N_T": 8,
        "incep_ker": [3,5,7,11],
        "groups": 8,
    }
    return config

def SimVP2_2D(in_shape, pred_n_frames_per_step):
    config = {
        # model
        "in_shape": in_shape,
        "hid_S": 16,
        "hid_T": 256,
        "N_S": 4,
        "N_T": 8,
        "incep_ker": [3,5,7,11],
        "groups": 8,
    }
    return config

def UNet_2D(in_shape, pred_n_frames_per_step):
    config={
        # model
    }
    return config

def UNetPlus_temp_2D(in_shape, pred_n_frames_per_step):
    config={
        # model
        "input_frames": in_shape[0],
        "base_filters": 32,
    }
    return config

config_map_2d = {
    "PredFormer": PredFormer_2D,
    "SimVP": SimVP_2D,
    "SimVP2": SimVP2_2D,
    "UNet": UNet_2D,
    "UNetPlus_temp": UNetPlus_temp_2D,
}

def PredFormer_3D(in_shape, pred_n_frames_per_step):
    config = { "model_config": {
        # model
        # image h w c
        "image_depth": in_shape[-3],
        'height': in_shape[-2],
        'width': in_shape[-1],
        'num_channels': in_shape[1],
        # video length in and out
        'pre_seq': in_shape[0],
        'after_seq': pred_n_frames_per_step,
        # patch size
        'patch_size': 8,
        'dim': 256, 
        'heads': 8,
        'dim_head': 32,
        # dropout
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'drop_path': 0.25,
        'scale_dim': 2,
        # depth
        'depth': 1,
        'Ndepth': 2, # For FullAttention-8, for BinaryST, BinaryST, FacST, FacTS-4, for TST,STS-3, for TSST, STTS-2
    }}
    return config

def SimVP_3D(in_shape, pred_n_frames_per_step):
    config = {
        # model
        "in_shape": in_shape,
        "hid_S": 16,
        "hid_T": 256,
        "N_S": 4,
        "N_T": 8,
        "incep_ker": [3,5,7,11],
        "groups": 8,
    }
    return config

def SimVP2_3D(in_shape, pred_n_frames_per_step):
    config = {
        # model
        "in_shape": in_shape,
        "hid_S": 16,
        "hid_T": 256,
        "N_S": 4,
        "N_T": 8,
        "incep_ker": [3,5,7,11],
        "groups": 8,
    }
    return config

def UNet_3D(in_shape, pred_n_frames_per_step):
    config={
        # model
    }
    return config

def UNetPlus_temp_3D(in_shape, pred_n_frames_per_step):
    config={
        # model
        "input_frames": in_shape[0],
        "base_filters": 16,
    }
    return config


config_map_3d = {
    "PredFormer": PredFormer_3D,
    "SimVP": SimVP_3D,
    "SimVP2": SimVP2_3D,
    "UNet": UNet_3D,
    "UNetPlus_temp": UNetPlus_temp_3D,
}