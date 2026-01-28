import torch
from hrnetpose_model import get_pose_net
from configs import load_configs

# 1. Load the YAML
cfg = load_configs('hrnet_w48_model_configs.yaml')

# 2. Initialize Model
model = get_pose_net(cfg, is_train=False)

# 3. Load Weights
checkpoint = torch.load('/home/clinton-mwangi/Downloads/pose_hrnet_w48.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

print("Model successfully initialized from YAML!")
