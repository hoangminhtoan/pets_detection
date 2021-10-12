import argparse
import cv2
import os
import numpy as np
import time
import argparse
import warnings

from numpy.core.shape_base import block
import torch
from tool.darknet2pytorch import Darknet
import glob

from datetime import datetime
warnings.filterwarnings("ignore")

CONSTANT_LIST = [i for i in range(0, 9, 2)] + [i for i in range(9, 24)] + [i for i in range(24, 32, 2)]
print(cv2.__version__)

def nms_cpu(boxes, confs, nms_thresh, min_mode=False):
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)

def post_processing(img, conf_thresh, nms_thresh, output):
    
    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):
       
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]
        
        bboxes = []
        # nms for each class
        for j in range(num_classes):
            
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
            
            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2],
                                   ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
        
        bboxes_batch.append(bboxes)
    
    return bboxes_batch

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=True):
    """
    Args: img (BCHW) - N torch tensor RGB images loaded from cv2"""
    model.eval()
    # if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
    #     # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    #     img = torch.from_numpy(img[..., ::-1].transpose(2, 0, 1)).float().div(255.0).unsqueeze(0) # BGR to RGB, BHWC to BCHW
    # elif type(img) == np.ndarray and len(img.shape) == 4:
    #     img = torch.from_numpy(img[..., ::-1].transpose(0, 3, 1, 2)).float().div(255.0) # BGR to RGB, BHWC to BCHW

    if use_cuda:
        img = img.to(device)
    
    with torch.no_grad():
        output = model(img)

    return post_processing(img, conf_thresh, nms_thresh, output)

def draw_box(frame, boxes, box_wh_min=20, box_wh_max=400):
    height, width, _ = frame.shape
    buffer = False
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = np.array(boxes)           
    if len(boxes) == 0:
        return buffer
    
    # convert boxes: xyxyn to xyxy
    boxes[:, :4:2] *= width
    boxes[:, 1:4:2] *= height
    
    boxes = boxes.astype(np.int16)
    
    for box in boxes:
        if box_wh_min <= box[2] - box[0] <= box_wh_max and box_wh_min <= box[3] - box[1] <= box_wh_max:
            # get class id
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), [0, 105, 204], thickness=3)
            buffer = True
    
    return buffer

def pet_detection(detector, block_images, handle_frame=16):
    if len(block_images) < handle_frame:
        print(f"Error: Number of continous frames is not enough for inference rule {len(block_images)}")
        return

    # sensitive = int(handle_frame * 0.3) # sensitive requires 30% satisfied frame in handle_frame block      
    # sensitive = int(handle_frame * 0.5) # sensitive requires 30% satisfied frame in handle_frame block      case 4
    sensitive = int(handle_frame * 0.25) # sensitive requires 30% satisfied frame in handle_frame block      case 6, 7
    # Choose 24/32 images for inference
    batch_images = [block_images[i] for i in CONSTANT_LIST]
    # batch_images = block_images
    # trigger_threshold = int((len(batch_images) - handle_frame)/2) + 1   # case 1, 4
    trigger_threshold = int((len(batch_images) - handle_frame)/2)   # case 2, 3, 6, 7
    # Resize batch_image
    input_size = (detector.width, detector.height)
    img = [cv2.resize(x, input_size) for x in batch_images]
    # Stack
    img = np.stack(img, 0)
    # Convert
    if type(img) == np.ndarray and len(img.shape) == 3:
        img = img[..., ::-1].transpose((2, 0, 1))   # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().div(255.0).unsqueeze(0) # uint8 to fp32, normalize 0 - 255 to 0.0 - 1.0, CHW to BCHW
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().div(255.0) # uint8 to fp32, normalize 0 - 255 to 0.0 - 1.0
    # Run batch detect        
    # outputs = batch_detect(detector=detector, resized_images=img, batch_images=batch_images, batch_size=16, THRESHOLD_DET=0.5)
    outputs, out_images = batch_detect(detector=detector, resized_images=img, batch_images=batch_images, batch_size=16, THRESHOLD_DET=0.5)
    if any(outputs):
        sum_list = np.array([sum(outputs[i:i+handle_frame]) for i in range(len(outputs)-handle_frame+1)])
        countTrigger = sum_list >= sensitive
        if np.all(countTrigger):
            # draw_images = [batch_images[0]]
            draw_images = [out_images[0]]
            return True, True, draw_images # pet_flag, tracking_flag, image with postive bbox
        elif sum(countTrigger) >= trigger_threshold:
            idxs = np.where(outputs)[0]
            # draw_images = [batch_images[idxs[0]]]
            draw_images = [out_images[idxs[0]]]
            return False, True, draw_images
        else:
            return False, False, None
    else:
        return False, False, None

def batch_detect(detector, resized_images, batch_images, batch_size, THRESHOLD_DET):
    # Split input by batch_size
    inputs = torch.split(resized_images, batch_size, dim=0)
    list_bboxes = [do_detect(model=detector, img=inputs[i], conf_thresh=THRESHOLD_DET, nms_thresh=0.4, use_cuda=True) for i in range(len(inputs))]
    # Flatten bbox per frame:
    frame_bbox = [item for sublist in list_bboxes for item in sublist]
    outputs = [draw_box(batch_images[i], frame_bbox[i]) for i in range(len(frame_bbox))]
    # return outputs
    return outputs, batch_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to folder video")
    parser.add_argument("--weights", default="yolov4.weights", help="yolo weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg", help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data", help="path to data file")
    args = parser.parse_args()
    print(args)
    # number of frame in a block
    block_of_frames = 32 # 32
    fps_target = 10
    # Config device
    device = torch.device('cuda:0')
    # Load model detection
    detector = Darknet(args.config_file, inference=True)
    detector.load_weights(args.weights)
    detector.to(device)
    torch.set_grad_enabled(False)

    count_time = 0
    total_time = 0
    
    paths = list(glob.glob(os.path.join(args.path, "*/*.mp4")))
    paths.sort()
    for path in paths:
        print(path)
        parent_folder = path.split('/')[-2]
        video_name = path.split('/')[-1]
        dir = os.path.join('results_tuanng_case_6_2', parent_folder, video_name)

        #load video
        video = cv2.VideoCapture(path)
        ret, img = video.read()
        if img is None:
            print("Read video error\n")
            continue
        # get frames at difference position in 1s
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == float("inf"):
            print("invalid video fps")
            fps = 10
        assert fps >= fps_target, f'fps must >= target fps'
        frame_step = fps // fps_target
        if (frame_step < 1) or np.isnan(frame_step):
            frame_step = 1
        print(f"VIDEO FPS = {fps} || FPS_TARGET = {fps_target} || FRAME_STEP = {frame_step}")
        # height, width, channels = img.shape
        print(img.shape)
        # LOCAL VARIABLE FOR while loop video process
        frame_id = -1
        block_images = []
        no_blocks_tracking = 3
        isTrigger = [False] * no_blocks_tracking
        while(video.isOpened()):
            frame_id += 1
            ret, img = video.read()
            if img is None:
                break
            if frame_id % frame_step == 0:
                start = time.time()
                block_images.append(img)
                if len(block_images) == block_of_frames:
                    # Run pet inference rule
                    pet_flag, tracking_flag, draw_images = pet_detection(detector=detector, block_images=block_images, handle_frame=16)
                    # Tracking block_images rule
                    isTrigger.append(tracking_flag)
                    isTrigger.pop(0)
                    if pet_flag or sum(isTrigger)>=2:
                        print("{} DETECTED PETTT!!!".format(time.strftime("%d-%m-%Y %H:%M:%S")))
                        # Save result
                        if draw_images[0].size:
                            pos_frame = time.strftime("%d%m%Y_%H_%M_%S")
                            imgname = f"frame_{pos_frame}_pet.jpg"
                            if not os.path.exists(dir):
                                os.makedirs(dir)
                            cv2.imwrite(f"{dir}/{imgname}", draw_images[0])
                            print(f"Result saved at {dir}")
                            break
                    block_images = []
                count_time += 1
                end = time.time()
                total_time += (end - start)
            if cv2.waitKey(24) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()
    print(f"FPS: {(count_time / total_time):.2f}")