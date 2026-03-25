import torch
import torchvision.models as tvm
import torch.nn as nn
import traceback

try:
    p = 'models/EfficientNet_v2_Phase1_best.pth'
    st = torch.load(p, map_location='cpu')
    m = tvm.efficientnet_b0(weights=None)
    
    # Custom classifier from Shilpa-Golla codebase
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 256, bias=True),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, 4, bias=True)
    )
    
    state_dict = st.get('model_state_dict', st.get('state_dict', st))
    missing, unexpected = m.load_state_dict(state_dict, strict=False)
    print("Missing:", missing)
    print("Unexpected:", unexpected)
    print("Loaded state dict successfully!")
except Exception as e:
    traceback.print_exc()
