import os
import torch
from . import encoder

__version__ = "1.0.0"

def get_model():
    """Create model."""
    cdir = os.path.dirname(__file__)
    model_path = "models/stylegan3_encoder.pth"
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = encoder.GradualStyleEncoder()
    # model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.load_state_dict(torch.load(checkpoint))

    model = model.eval()
    return model
