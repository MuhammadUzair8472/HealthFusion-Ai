import torch
p = 'models/EfficientNet_v2_Phase1_best.pth'
st = torch.load(p, map_location='cpu')
sd = st.get('model_state_dict', st.get('state_dict', st))
for k, v in sd.items():
    if k.startswith('classifier'):
        print(f"{k}: {v.shape}")
