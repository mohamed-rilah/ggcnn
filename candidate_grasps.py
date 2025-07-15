import cv2
import torch 
import numpy as np 
import matplotlib.pyplot as plt

from models.ggcnn import GGCNN

def image_preprocessing(image_path = 'box_depthim.png'): 
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if depth_image.ndim == 3: 
        depth_image = depth_image[:, :, 0]

    normalised_depth = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
    resized_depth = cv2.resize(normalised_depth, (224, 224))
    tensor_depth = torch.tensor(resized_depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor_depth

def get_candidate_grasp(depth_image, model): 
    tensor_depth = image_preprocessing(depth_image)

    with torch.no_grad(): 
        output = model(tensor_depth)

    return output

model_path = 'external-repos/ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'

model = GGCNN()
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

depth_image_path = 'Images/box_depthim.png'

grasp = get_candidate_grasp(depth_image_path, model)
print(f'Output from model: ', {grasp})

print(grasp[0].shape)

pos_output = grasp[0].squeeze().cpu().numpy()
print(f'Shape of tensor 0: {pos_output.shape}')
max_index = np.argmax(pos_output)
y, x = np.unravel_index(max_index, pos_output.shape)
print(f'Grasp Centre: (x={x}, y={y})')

plt.imshow(pos_output, cmap='hot')
plt.title('Heatmap for proposed grasp centre point(s)')
plt.colorbar()
plt.show()

depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)