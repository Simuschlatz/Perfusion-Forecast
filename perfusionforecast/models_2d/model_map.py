from .simvp import SimVP
from .simvp2 import SimVP_Model as SimVP2
from .unetplus_temp import UNet2DPlusTemporal
from .lightning_model import Pl_Model
from .unet import UNet3D as UNet
from .predformer import PredFormer_Model as PredFormer

model_map={
    "SimVP": SimVP,
    "SimVP2": SimVP2,
    "UNetPlus_temp": UNet2DPlusTemporal,
    "UNet": UNet,
    "PredFormer": PredFormer,
}