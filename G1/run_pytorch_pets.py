import argparse
import cv2
import os
import numpy as np
import pandas as pd
import time
from PIL import Image
import argparse
import torch
from torchvision import transforms
import torch.nn.functional as F
from tool.darknet2pytorch import Darknet
import glob

from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, confusion_matrix

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


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        
def draw_box(frame, boxes, box_wh_min=0, box_wh_max=400):
    
    buffer = []
    conf_list = []
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = np.array(boxes)           
    if len(boxes) == 0:
        return buffer, frame, conf_list
    
    # convert boxes: xyxyn to xyxy
    boxes[:, :4:2] *= width
    boxes[:, 1:4:2] *= height
    
    # boxes = boxes.astype(np.int16)
    
    for box in boxes:
        if box_wh_min <= box[2] - box[0] <= box_wh_max and box_wh_min <= box[3] - box[1] <= box_wh_max:
            # get class id
            if box[-1] == 0 or box[-1] == 1:
                color = (0, 0, 255)
                label = 'pet - ' + str(round(box[-2], 2))
                buffer.append(True)
            else:
                color = (0, 255, 0)
                label = 'human - ' + str(round(box[-2], 2))
                buffer.append(False)
            conf_list.append(box[-2])
            box = box.astype(np.int16)
            plot_one_box((box[0], box[1], box[2], box[3]), frame, color, label)
    
    return buffer, frame, conf_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to folder video")
    parser.add_argument("--crop_glassArea", type=str, help="Set Variable to Crop Glass Area or Not? True:Crop or False:Not Crop")
    args = parser.parse_args()

    # Configure hyperparameters
    THRESHOLD_DET = 0.5 #0.65
    #input_size = (608, 608)
    input_size = (736, 416)
    detector_bs = 10

    fps_target = 10
    handle_frame = fps_target * 2
    cover_threshold = 3 #handle_frame * 0.3
    trigger_threshold = 3
    trigger_reset_threshold = fps_target * 5
    trigger_time = fps_target * 0
    
    preds = []
    labels = []
    names = []

    device = torch.device('cuda:0')
    label_mapping = ["normal", "coverface"]

    # Load model detection
    weightfile = "/vinbrain/thanh/SangEm/pets_project/deploy_team_new/20211010_weights/yolov4-custom-3-classes_final.weights"
    cfgfile = "/vinbrain/thanh/SangEm/pets_project/deploy_team_new/20211010_weights/yolov4-custom-3-classes.cfg"
    detector = Darknet(cfgfile, inference=True)
    detector.load_weights(weightfile)
    detector.to(device)
    torch.set_grad_enabled(False)

    count_time = 0
    total_time = 0
    
    # create folder to write video
    folder2write = "211006_model1010"
    
    # paths = os.listdir(args.path)
    paths = list(glob.glob(os.path.join(args.path, "*/*/*/*.mp4")))
    paths.sort()
    for path in paths:

        print(path)
        folder_cats = path.split('/')[-4]
        folder = path.split('/')[-3]
        video_name = path.split('/')[-1][:-4]
#         video = cv2.VideoCapture(os.path.join(args.path, path))
        video = cv2.VideoCapture(path)
        
        ret, img = video.read()
        if img is None:
            print("Read video error\n")
            continue
            
        if args.crop_glassArea=='True':
            print("crop_glassArea")
            img = img[:, :crop_area]
        
        # get frames at difference position in 1s
        fps = video.get(cv2.CAP_PROP_FPS)
        # if fps == float("inf"):
            # print("invalid video fps")
            # fps = 30
        # assert fps >= fps_target  # fps must >= target fps
        # targets_ids = set((np.arange(fps_target) * fps / fps_target).astype(int))
        
        if fps == float("inf") or fps < fps_target:
            print("invalid video fps")
            fps = 30
        
        frame_step = fps // fps_target
        
        height, width, channels = img.shape
        print(img.shape)
        images = []
        images_resized = []
        images_vis = []
        # conf_images_vis = []
        dict_idx_conf = {}
        conds = [False] * handle_frame 
        countTrigger = 0
        isTrigger = False
        frame_id = 0
        size = (width, height)
        predict = False

        while ret:
            ret, img = video.read()
            if img is None:
                break

            frame_id += 1

            if frame_id % frame_step == 0:
                start = time.time()
                if args.crop_glassArea=='True':
                    img = img[:, :crop_area]
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
                        buffer, img_vis, conf_list = draw_box(images[i], boxes[i])
                        # conf_images_vis.append(max(conf_list))
                        images_vis.append(img_vis)
                        if len(conf_list) > 0:
                            dict_idx_conf[i] = max(conf_list)
                        # store predict of frame
                        conds.append(True if True in buffer else False)
                    
                    for i in range(1, detector_bs+1):
                        # check rule base, true if catch coverface
                        if sum(conds[i:handle_frame + i]) >= cover_threshold:
                            resetTrigger = 0
                            if isTrigger == False:
                                predict = True
                                print(f"Pet at the {((frame_id - (detector_bs - i)) / fps):.1f}th seconds")
                                if not os.path.exists(os.path.join(folder2write, folder_cats, folder, video_name)):
                                    os.makedirs(os.path.join(folder2write, folder_cats, folder, video_name))
                                for idx_vis, conf in sorted(dict_idx_conf.items(), key=lambda x: x[1], reverse=True)[:1]:
                                    idx = int((frame_id - (detector_bs - idx_vis)) / fps)
                                    cv2.imwrite(os.path.join(folder2write,
                                                             folder_cats,
                                                             folder,
                                                             video_name,
                                                             f'video_name_{idx}_{conf}.jpg'), images_vis[idx_vis])
                                isTrigger = True
                                countTrigger = 0
                                # break
                        else:
                            if isTrigger:
                                resetTrigger += 1
                                if resetTrigger > trigger_reset_threshold:
                                    conds = [False] * (handle_frame + detector_bs)
                                    resetTrigger = 0
                                    isTrigger = False

                    conds = conds[detector_bs:]
                    # if isTrigger:
                    #    break
                    images_resized = []
                    images = []
                    images_vis = []
                count_time += 1
                end = time.time()
                total_time += (end - start)
        print()
#         with open("sametime_2021106_model0510.txt", 'a') as fn:
#             fn.write(path + ' ')
#             fn.write("True " if path.find("pos") > -1 else "False ")
#             fn.write(str(predict) + '\n')
        
        names.append(path)
        labels.append(True if path.find("pos") > -1 else False)
        preds.append(predict)
        
    print(f"FPS: {(count_time / total_time):.2f}")
    f1 = fbeta_score(y_true=labels, y_pred=preds, beta=1, pos_label=1, average='binary')

    precision = precision_score(y_true=labels, y_pred=preds, pos_label=1, average='binary')
    recall = recall_score(y_true=labels, y_pred=preds, pos_label=1, average='binary')
    print(f1, precision, recall, THRESHOLD_DET)
    print(confusion_matrix(labels, preds))
    df = pd.DataFrame({'names': names, 'labels': labels, 'preds': preds}).to_csv("211006_model1010_%f.csv"%THRESHOLD_DET, index=False)