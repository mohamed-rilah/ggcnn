import cv2
import torch 
import numpy as np 
import matplotlib.pyplot as plt

import time

from models.ggcnn import GGCNN

def image_preprocessing(image_path = 'cube.png'): 
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if depth_image.ndim == 3: 
        depth_image = depth_image[:, :, 0]

    normalised_depth = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
    tensor_depth = torch.tensor(normalised_depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor_depth

def get_candidate_grasp(depth_image, model): 
    tensor_depth = image_preprocessing(depth_image)

    with torch.no_grad(): 
        output = model(tensor_depth)

    return output

model_path = 'external-repos/ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'

model_start = time.time()

model = GGCNN()
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

depth_image_path = 'Images/cube.png'

grasp = get_candidate_grasp(depth_image_path, model)

pos_output = grasp[0].squeeze().cpu().numpy()
cos_output = grasp[1].squeeze().cpu().numpy()
sin_output = grasp[2].squeeze().cpu().numpy()
width_output = grasp[3].squeeze().cpu().numpy()

max_index = np.argmax(pos_output)
y, x = np.unravel_index(max_index, pos_output.shape)

model_end = time.time()

angle_radians = np.arctan2(sin_output[y,x], cos_output[y,x]) / 2.0
angle_degrees = np.degrees(angle_radians)

print(f'Model Runtime: {model_end-model_start} seconds')

print(f'Grasp Centre: (x={x}, y={y})')
print(f'Cos output @ Grasp Centre {cos_output[y,x]}')
print(f'Sin output @ Grasp Centre {sin_output[y,x]}')
print(f'Gripper Width @ Grasp Centre {width_output[y,x]}')
print(f'Computed angle output @ Grasp Centre {angle_radians} radians / {angle_degrees} degrees')

plt.imshow(pos_output, cmap='hot')
plt.title('Heatmap for proposed grasp centre point(s)')
plt.colorbar()
plt.show()

depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

dx = np.cos(angle_radians) * 30
dy = np.sin(angle_radians) * 30 

orientation_x1 = x - dx
orientation_y1 = y - dy
orientation_x2 = x + dx
orientation_y2 = y + dy

plt.figure(figsize=(6,4))
plt.imshow(depth_image, cmap='grey')
plt.imshow(pos_output, cmap='hot', alpha=0.45)
plt.colorbar(label='Prediction')

plt.plot([orientation_x1, orientation_x2], [orientation_y1, orientation_y2], color='cyan', linewidth=2)
plt.plot(x, y, 'bo')

plt.title('Visualised Heatmap and Depth Image')
plt.axis('off')
plt.show()