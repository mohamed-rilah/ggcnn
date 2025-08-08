import cv2
import torch 
import numpy as np 
import matplotlib.pyplot as plt

import time

from models.ggcnn import GGCNN

def image_preprocessing(image_path): 
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if depth_image.ndim == 3: 
        depth_image = depth_image[:, :, 0]

    normalised_depth = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
    tensor_depth = torch.tensor(normalised_depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor_depth

def get_candidate_grasp(depth_image, model): 
    tensor_depth = image_preprocessing(depth_image)

    start_time = time.time()

    with torch.no_grad(): 
        output = model(tensor_depth)

    end_time = time.time()

    duration = end_time - start_time
    print(f'Model Duration: {duration:.4f} seconds')

    return output

def visualise_top_grasps(depth_image_path, model, n): 

    grasp = get_candidate_grasp(depth_image_path, model)

    pos_output = grasp[0].squeeze().cpu().numpy()
    cos_output = grasp[1].squeeze().cpu().numpy()
    sin_output = grasp[2].squeeze().cpu().numpy()
    width_output = grasp[3].squeeze().cpu().numpy()

    top_n_indices = np.argsort(pos_output.flatten())[::-1][:n]

    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    rows = (n // 2) + (n % 2)
    cols = 2
    
    for i, idx in enumerate(top_n_indices):
        y, x = np.unravel_index(idx, pos_output.shape)

        angle_radians = np.arctan2(sin_output[y,x], cos_output[y,x]) / 2.0
        angle_degrees = np.degrees(angle_radians)

        print(f'\nGrasp {i+1} / {n}')
        print(f'Grasp Centre: (x={x}, y={y})')
        print(f'Cos output @ Grasp Centre {cos_output[y,x]}')
        print(f'Sin output @ Grasp Centre {sin_output[y,x]}')
        print(f'Gripper Width @ Grasp Centre {width_output[y,x]}')
        print(f'Computed angle output @ Grasp Centre {angle_radians} radians / {angle_degrees} degrees')

        grasp_depth = depth_image[y,x]
        scale_factor = grasp_depth / np.max(depth_image)
        dx = np.cos(angle_radians) * 30 * scale_factor
        dy = np.sin(angle_radians) * 30 * scale_factor

        orientation_x1 = x - dx
        orientation_y1 = y - dy
        orientation_x2 = x + dx
        orientation_y2 = y + dy

        plt.subplot(rows, cols, i + 1)

        plt.imshow(depth_image, cmap='grey')
        plt.imshow(pos_output, cmap='hot', alpha=0.45)

        plt.plot([orientation_x1, orientation_x2], [orientation_y1, orientation_y2], color='cyan', linewidth=2)
        plt.plot(x, y, 'bo')

        plt.title(f'Visualising Grasp {i+1}/{n}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

model_path = 'external/ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'

model = GGCNN()
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

depth_image_path = 'images/cube.png'

visualise_top_grasps(depth_image_path, model, n=5)