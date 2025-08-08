import torch

from models.ggcnn import GGCNN

model_path = 'external/ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'

model = GGCNN()
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully!")
print(model)