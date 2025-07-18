import cv2
import torch 
import numpy as np 
import matplotlib.pyplot as plt

import time

from models.ggcnn import GGCNN

def image_preprocessing(image_path = 'box_depthim.png'): 
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if depth_image.ndim == 3: 
        depth_image = depth_image[:, :, 0]

    normalised_depth = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
    resized_depth = cv2.resize(normalised_depth, (300, 300))
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

start = time.time()

grasp = get_candidate_grasp(depth_image_path, model)

pos_output = grasp[0].squeeze().cpu().numpy()
cos_output = grasp[1].squeeze().cpu().numpy()
sin_output = grasp[2].squeeze().cpu().numpy()
width_output = grasp[3].squeeze().cpu().numpy()

max_index = np.argmax(pos_output)
y, x = np.unravel_index(max_index, pos_output.shape)

end = time.time()

print(f'Code Runtime: {end-start} seconds')

print(f'Grasp Centre: (x={x}, y={y})')
print(f'Cos output @ Grasp Centre {cos_output[y,x]}')
print(f'Sin output @ Grasp Centre {sin_output[y,x]}')
print(f'Gripper Width @ Grasp Centre {width_output[y,x]}')


plt.imshow(pos_output, cmap='hot')
plt.title('Heatmap for proposed grasp centre point(s)')
plt.colorbar()
plt.show()

depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

scaled_pos_output = cv2.resize(pos_output, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_LINEAR)

scaled_height = scaled_pos_output.shape[0] / pos_output.shape[0] 
scaled_width = scaled_pos_output.shape[1] / pos_output.shape[1]

scaled_x = int(x * scaled_width) 
scaled_y = int(y * scaled_height)
print(f'Scaled Grasp Centre: (x={scaled_x}, y={scaled_y})')

plt.figure(figsize=(6,4))
plt.imshow(depth_image, cmap='grey')
plt.imshow(scaled_pos_output, cmap='hot', alpha=0.45)
plt.colorbar(label='Prediction')
plt.title('Visualised Heatmap and Depth Image')
plt.axis('off')
plt.show()