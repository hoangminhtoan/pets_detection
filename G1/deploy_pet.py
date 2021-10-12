import argparse
import cv2
import os
import numpy as np
import time
from PIL import Image
import argparse
import torch
from torchvision import transforms
import torch.nn.functional as F
from tool.darknet2pytorch import Darknet
import glob

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
    model.eval()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)

    if use_cuda:
        img = img.to(device)
    
    with torch.no_grad():
        output = model(img)

    return post_processing(img, conf_thresh, nms_thresh, output)


def draw_box(frame, boxes, box_wh_min=20, box_wh_max=400):
    
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
            buffer = True
    
    return buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to folder video")
    parser.add_argument("--crop_glassArea", type=str, help="Set Variable to Crop Glass Area or Not? True:Crop or False:Not Crop")
    args = parser.parse_args()

    # Configure hyperparameters
    THRESHOLD_DET = 0.65
    input_size = (608, 608)
    detector_bs = 10

    fps_target = 10
    handle_frame = fps_target * 2
    cover_threshold = 11 #handle_frame * 0.3
    trigger_threshold = 3
    trigger_reset_threshold = fps_target * 5
    trigger_time = fps_target * 0

    device = torch.device('cuda:0')
    label_mapping = ["normal", "coverface"]

    # Load model detection
    weightfile = "pets.weights"
    cfgfile = "pets.cfg"
    detector = Darknet(cfgfile, inference=True)
    detector.load_weights(weightfile)
    detector.to(device)
    torch.set_grad_enabled(False)

    count_time = 0
    total_time = 0
    
#     paths = os.listdir(args.path)
    paths = list(glob.glob(os.path.join(args.path, "*/*.mp4")))
    paths.sort()
    for path in paths:

        print(path)
        parent_folder = path.split('/')[-2]
        video_name = path.split('/')[-1]
        dir = os.path.join('results', parent_folder, video_name)
#         video = cv2.VideoCapture(os.path.join(args.path, path))
        video = cv2.VideoCapture(path)
        
        ret, img = video.read()
        if img is None:
            print("Read video error\n")
            continue
            
        if args.crop_glassArea=='True':
            print("crop_glassArea")

        
        # get frames at difference position in 1s
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == float("inf"):
            print("invalid video fps")
            fps = 30
        assert fps >= fps_target  # fps must >= target fps
        targets_ids = set((np.arange(fps_target) * fps / fps_target).astype(int))
        
        height, width, channels = img.shape
        print(img.shape)
        images = []
        images_resized = []
        conds = [False] * handle_frame 
        countTrigger = 0
        isTrigger = False
        frame_id = 0
        size = (width, height)

        while ret:
            ret, img = video.read()
            if img is None:
                break

            frame_id += 1

            if (frame_id % fps) in targets_ids:
                start = time.time()
                img_resized = cv2.resize(img, input_size)
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                images_resized.append(img_resized)
                images.append(img)

                if len(images) == detector_bs:
                    # detect faces in many images
                    images_resized = np.stack(images_resized)

                    # boxes: xyxyn
                    boxes = do_detect(detector, images_resized, conf_thresh=THRESHOLD_DET,
                                      nms_thresh=0.4, use_cuda=True)

                    for i in range(detector_bs):
                        # each frame contain many faces or 0 faces
                        buffer = draw_box(images[i], boxes[i])
                        
                        # store predict of frame
                        conds.append(True == buffer) 
                    
                    for i in range(1, detector_bs+1):
                        # check rule base, true if catch coverface
                        if sum(conds[i:handle_frame + i]) >= cover_threshold:
                            resetTrigger = 0
                            if isTrigger == False:
                                countTrigger += 1
                                if countTrigger >= trigger_threshold:
                                    if not os.path.exists(os.path.join(dir, video_name)):
                                        os.makedirs(os.path.join(dir, video_name))
                                    print(f"Pet at the {((frame_id - (detector_bs - i)) / fps):.1f}th seconds")
                                    idx = int(((frame_id - (detector_bs - i)) / fps) % fps)
                                    cv2.imwrite(os.path.join(dir, '{:07d}.jpg'.format(idx)), images[idx])
                                    isTrigger = True
                                    countTrigger = 0
#                                     break
                        else:
                            if isTrigger:
                                resetTrigger += 1
                                if resetTrigger > trigger_reset_threshold:
                                    conds = [False] * (handle_frame + detector_bs)
                                    resetTrigger = 0
                                    isTrigger = False

                    conds = conds[detector_bs:]
#                     if isTrigger:
#                         break
                    images_resized = []
                    images = []
                count_time += 1
                end = time.time()
                total_time += (end - start)
        print()
    print(f"FPS: {(count_time / total_time):.2f}")