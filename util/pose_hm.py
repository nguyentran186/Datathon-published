import cv2
import numpy as np
from ultralytics import YOLO
from skimage.transform import resize


cell_size = 20

def increase_values_around_point(image, center, radius, increment_value=1):
    """
    Increase pixel values around a specific point in the image.

    Parameters:
    - image: NumPy array representing the image.
    - center: Tuple (x, y) specifying the center of the circular region.
    - radius: Radius of the circular region.
    - increment_value: Value to increment pixel values (default is 1).

    Returns:
    - Modified image array.
    """
    # Create a meshgrid of coordinates
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    # Calculate distances from each point to the center
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Create a binary mask for the circular region
    circular_mask = distances <= radius

    # Increase pixel values within the circular region
    image[circular_mask] += increment_value

    return image

def get_row_col(x, y):
    row = int(y/cell_size)
    col = int(x/cell_size)
    return row, col

def draw_grid(image):
    vid_shape = image.shape
    width, height = vid_shape[1], vid_shape[0]

    n_cols = int(width/cell_size)   
    n_rows = int(height/cell_size) 
    for i in range(n_rows):
        start_point = (0, (i+1)*cell_size)
        end_point = (width, (i+1)*cell_size)
        color = (255,255,255)
        thickness = 1
        image = cv2.line(image, start_point,end_point,color,thickness)

    for i in range(n_cols):
        start_point = ((i+1)*cell_size, 0)
        end_point = ((i+1)*cell_size, height)
        color = (255,255,255)
        thickness = 1
        image = cv2.line(image, start_point,end_point,color,thickness)

    return image

def heatmap_im(heatmap, temp_im, height, width, alpha):
    temp_heat = heatmap.copy()
    temp_heat = resize(temp_heat, (height,width))
    temp_heat = temp_heat/np.max(temp_heat)
    temp_heat = np.uint8(temp_heat*200)

    image_heat = cv2.applyColorMap(temp_heat, cv2.COLORMAP_JET)
    cv2.addWeighted(image_heat, alpha, temp_im, 1-alpha, 0, temp_im)
    _, jpeg = cv2.imencode('.jpg', temp_im)
    return jpeg

def pose_heatmap(vid):
    model = YOLO('yolov8m-pose.pt')
    cap = cv2.VideoCapture(vid)

    first_im = cap.read()[1]
    vid_shape = first_im.shape
    width, height = vid_shape[1], vid_shape[0]

    cell_size = 20
    n_cols = int(width/cell_size)   
    n_rows = int(height/cell_size) 
    alpha = 0.4

    heat_matrix = np.zeros((n_rows, n_cols, 3))
    kneel_matrix = np.zeros((n_rows, n_cols, 3))
    high_matrix = np.zeros((n_rows, n_cols, 3))
    stand_matrix = np.zeros((n_rows, n_cols, 3))
    frame_index = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        frame_index = frame_index + 30
        success, frame = cap.read()
        if success:
            ####### HEAT MATRIX GENERATOR
            results = model(frame, conf=0.2)
            boxes = results[0].boxes.xyxy.squeeze()
            pred_boxes = results[0].boxes
            if boxes.dim() == 1:
                d = pred_boxes[0]
                c = int(d.cls)
                item = boxes
                x1,y1,x2,y2 = int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())
                p1, p2 = get_row_col(x1,y1), get_row_col(x2,y2)   
                heat_matrix = increase_values_around_point(heat_matrix, (int((p1[1]+p2[1])/2), int((p1[0]+p2[0])/2)), 3)
            if boxes.dim() > 1:
                for c, item in enumerate(boxes):
                    x1,y1,x2,y2 = int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())
                    p1, p2 = get_row_col(x1,y1), get_row_col(x2,y2)  
                    heat_matrix = increase_values_around_point(heat_matrix, (int((p1[1]+p2[1])/2), int((p1[0]+p2[0])/2)), 3)

            result_keypoint = results[0].keypoints.xyn.cpu().numpy()[0]
            detect_list = []
            #if detect only 1 
            if boxes.dim() == 1:
                item = boxes
                detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])
            ##more than 1 
            else:
                for item in boxes:
                    detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])

            
            lb_list=[]
            if len(detect_list)>0:
                for ind, item in enumerate(detect_list):
                    rshoulder , lshoulder, rleg, lleg = result_keypoint[6],result_keypoint[5],result_keypoint[12],result_keypoint[11]
                    diff = [rshoulder[1]-rleg[1], lshoulder[1]-lleg[1]]

                    x1,y1,x2,y2 = item
                    p1, p2 = get_row_col(x1,y1), get_row_col(x2,y2)

                    if (x1 < 800 or y2 > 500):
                        #### Kneel
                        if (y1<200): 
                            for i in range(p1[0], p1[0]+3):
                                for j in range(p1[1], p2[1]):
                                    high_matrix[i][j]+=1
                        elif (diff[0]<-0.07 or diff[1]<-0.07):
                            kneel_matrix = increase_values_around_point(kneel_matrix, (int((p1[1]+p2[1])/2), int((p1[0]+p2[0])/2)), 3)
                        else:
                            stand_matrix = increase_values_around_point(stand_matrix, (int((p1[1]+p2[1])/2), int((p1[0]+p2[0])/2)), 3)

        else:
            break
    temp_1 = first_im.copy()
    dense_im = heatmap_im(heat_matrix, temp_1, height, width, alpha)
    temp_2 = first_im.copy()
    high_im = heatmap_im(high_matrix, temp_2, height, width, alpha)
    temp_3 = first_im.copy()
    kneel_im = heatmap_im(kneel_matrix, temp_3, height, width, alpha)
    temp_4 = first_im.copy()
    stand_im = heatmap_im(stand_matrix, temp_4, height, width, alpha)

    return dense_im, high_im, kneel_im, stand_im