import cv2
import numpy as np
from ultralytics import YOLO
from skimage.transform import resize

cell_size = 20

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

def distribution_heatmap(vid):
    model = YOLO('yolov8m.pt')
    cap = cv2.VideoCapture(vid)

    first_im = cap.read()[1]
    vid_shape = first_im.shape
    width, height = vid_shape[1], vid_shape[0]

    cell_size = 20
    n_cols = int(width/cell_size)   
    n_rows = int(height/cell_size) 
    alpha = 0.4

    heat_matrix = np.zeros((n_rows, n_cols, 3))
    frame_index = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        frame_index = frame_index + 60
        success, frame = cap.read()
        if success:
            results = model(frame, conf=0.2)
            boxes = results[0].boxes.xyxy.squeeze()
            pred_boxes = results[0].boxes
            if boxes.dim() == 1:
                d = pred_boxes[0]
                c = int(d.cls)
                if results[0].names[c] != 'person': continue
                item = boxes
                x1,y1,x2,y2 = int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())
                p1, p2 = get_row_col(x1,y1), get_row_col(x2,y2)
                for i in range(p1[0], p2[0]):
                    for j in range(p1[1], p2[1]):
                        heat_matrix[i][j]+=1    
            if boxes.dim() > 1:
                for c, item in enumerate(boxes):
                    if results[0].names[c] != 'person': continue
                    x1,y1,x2,y2 = int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())
                    p1, p2 = get_row_col(x1,y1), get_row_col(x2,y2)
                    for i in range(p1[0], p2[0]):
                        for j in range(p1[1], p2[1]):
                            heat_matrix[i][j]+=1    
        else:
            break
    temp_heat = heat_matrix.copy()
    temp_heat = resize(temp_heat, (height,width))
    temp_heat = temp_heat/np.max(temp_heat)
    temp_heat = np.uint8(temp_heat*255)

    image_heat = cv2.applyColorMap(temp_heat, cv2.COLORMAP_JET)
    cv2.addWeighted(image_heat, alpha, first_im, 1-alpha, 0, first_im)
    _, jpeg = cv2.imencode('.jpg', first_im)
    return jpeg