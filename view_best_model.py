import torch
from model import CNNTransformer

model = CNNTransformer()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
print(model)