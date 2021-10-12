from ctypes import *
import os
import cv2
import argparse
import glob
from datetime import datetime
import numpy as np
import torch 
import torchvision
from tool.darknet2pytorch import Darknet
import torch.nn as nn
import collections

from configs import config
from utils import xywh2xyxy, xyxy2xywh, scale_coords

dt_string = datetime.now().strftime("%Y%m%d")

class Engine():
    def __init__(self, opt):
        self.fps_target = 10
        self.frame_step = 1
        self.duration = 0
        
        # Create folder to save result videos
        parent_folder = opt.source_url.strip().split('/')[-3].replace('\n', '') 
        folder_name = opt.source_url.strip().split('/')[-2].replace('\n', '')            
        if not os.path.exists(os.path.join(config.RESULT_DIR, f'{dt_string}_videos', f'{parent_folder}', f'{folder_name}', 'frames')):
            os.makedirs(os.path.join(config.RESULT_DIR, f'{dt_string}_videos', f'{parent_folder}', f'{folder_name}', 'frames'))
        
        
        self.videos_dir = os.path.join(config.RESULT_DIR, f'{dt_string}_videos', f'{parent_folder}', f'{folder_name}')
        self.frames_dir = os.path.join(config.RESULT_DIR, f'{dt_string}_videos', f'{parent_folder}',  f'{folder_name}', 'frames')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def nsm_cpu(self, boxes, confs, nms_thresh, min_mode=False):
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
    
    def intersec(self, bb1, bb2):
        ''' Get IOU ratio between two input box
        Args:
            bb1 (array): a list size of [x0, y0, width, height]
            bb2 (array): a list size of [x0, y0, width, height]
        Returns:
            return ration iou between two input box'''
        x_left = max(bb1[0], bb2[0])
        y_left = max(bb1[1], bb2[1])
        x_right = min(bb1[0] + bb1[2], bb2[0] + bb2[2]) 
        y_right = min(bb1[1] + bb1[3], bb2[1] + bb2[3])

        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_right - y_left)

        # compute the area of both AABBs
        bb1_area = bb1[2] * bb1[3]
        bb2_area = bb2[2] * bb2[3]
        return intersection_area / float(bb1_area + bb2_area - intersection_area)

    def post_processing(self, img, conf_thresh, nms_thresh, output):
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

                keep = self.nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
                
                if (keep.size > 0):
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2],
                                    ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
            bboxes_batch.append(bboxes)
        
        return bboxes_batch

    def detect(self, opt):
        import darknet
        #load detector model
        self.model, self.names, self.colors = darknet.load_network(opt.config_file, opt.data_file, opt.weights, batch_size=1)

        #load video
        cap = cv2.VideoCapture(opt.source_url)
        # Set resolution
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == float("inf"):
            print("invalid video fps")
            fps = 30
            
        if fps >= self.fps_target:
            self.frame_step = round(fps / self.fps_target)
        print(f'input video {opt.source_url} with frame rate: {fps} => skips {self.frame_step} frame(s)')
            
        video_width = 1920 # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = 1080 # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        darknet_width = darknet.network_width(self.model)
        darknet_height = darknet.network_height(self.model)
            
        # SEt output video
        video_name = opt.source_url.strip().split('/')[-1].replace('\n', '')
        if not os.path.exists(os.path.join(self.frames_dir, video_name)):
            os.makedirs(os.path.join(self.frames_dir, video_name)) 
        out = cv2.VideoWriter(os.path.join(self.frames_dir, video_name, f'{video_name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (video_width, video_height))
        frame_idx = -1
        offset = 0
        next_offset = 0
        images_list = []
        tmp_idx_conf = {}
        previous_box_list = []
        idx = 0
        FLAG_SAVE = False
        FLAG_DUPLICATE = False
            
        while cap.isOpened():
            frame_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_step == 0:
                self.duration += 1
                frame_resized = cv2.resize(frame, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
                img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
                darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
                    
                dets = darknet.detect_image(self.model, self.names, img_for_detect, thresh=opt.thresh)
                
                
                detections_adjusted = []
                max_conf = -100.0
                for label, conf, bbox in dets:
                    bbox_adjusted = self.convert_to_original(frame, darknet_height, darknet_width, bbox)
                    x0, y0, box_width, box_height = bbox_adjusted
                    #if label in ['cat', 'dog', 'human']: # and ((box_height > 48 and box_height < 128) or (box_width > 48 and box_width)):
                    if label in ['cat', 'dog'] and (box_height > 48 and box_width > 48):
                        max_conf = max(max_conf, float(conf))
                    box = [x0, y0, box_width, box_height]
                    # Checking duplicated bounding box in current frame with the previous frame
                    if len(previous_box_list) == 0: # initialize list of bounding b
                        previous_box_list.append(box)
                        detections_adjusted.append((str(label), conf, bbox_adjusted))
                    else:
                        for b in previous_box_list:
                            if self.intersec(b, box) > 0.7:
                                #print(b, box)
                                FLAG_DUPLICATE = True
                                break
                            FLAG_DUPLICATE = False
                        if FLAG_DUPLICATE is False: 
                            previous_box_list.append(box)
                            detections_adjusted.append((str(label), conf, bbox_adjusted))
                            if len(previous_box_list) > 4:
                                previous_box_list.pop(0)
                    # End checking duplicated bounding box in current frame with the priveious frame
                    FLAG_SAVE = True
                    print(previous_box_list, detections_adjusted)
                    # The bounding box should not be too close to the edge of frame
                    if (x0 > 10 and x0 < video_width - 10) and (y0 > 10 and y0 < video_height - 10):
                        frame = darknet.draw_boxes(detections_adjusted, frame, self.colors)

                if FLAG_SAVE:
                    FLAG_SAVE = False
                    images_list.append(frame)
                    if next_offset == 0 or next_offset == offset: # ignore 10 offsets after current detected offset
                        next_offset = offset + 10

                    tmp_idx_conf[idx] = max_conf
                    idx += 1

            if self.duration == self.fps_target * 2: # duration là  20 frames 
                if (next_offset == offset + 10) and len(images_list) > 2:
                    #now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    now = datetime.now().strftime("%Y%m%d")
                    print("[Warning!] at time {} Pet(s) were detected!".format(now))
                    #if not os.path.exists(os.path.join(self.frames_dir, video_name)):
                    #    os.makedirs(os.path.join(self.frames_dir, video_name)) 
                    sorted_tm_idx = dict(sorted(tmp_idx_conf.items(), key=lambda item: item[1], reverse=True))

                    if max_conf > 0:
                        prefix = '{:07}.jpg'
                        d = list(sorted_tm_idx.keys())[0] # get frame index that has the largest confidence score
                        cv2.putText(images_list[d], "key frame", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.imwrite(os.path.join(self.frames_dir, video_name, f'{offset}_{now}_{video_name}_{prefix.format(d)}'), images_list[d])
                # Reset variable
                self.duration = 0
                images_list = []
                tmp_idx_conf = {}
                offset+=1
                idx = 0
                
            # Write Video
            out.write(frame)
                
                # Show prediction
                # cv2.imshow(WINDOW_NAME, frame)
                
                #if cv2.waitKey(24) & 0xFF == ord('q')
                #   break
        cap.release()
            # cv2.destroyAllWindows()

    def detect_frame(self, opt):
        import darknet
        #load detector model
        self.model, self.names, self.colors = darknet.load_network(opt.config_file, opt.data_file, opt.weights, batch_size=1)

        #load video
        image_paths =  list(glob.glob(os.path.join(opt.source_url, "*.jpg")))
        image_paths.sort()
        for idx, image_path in enumerate(image_paths):
            #print(f'Image {idx+1}/{len(image_paths)}\n')
            if not os.path.isfile(image_path):
                print(f"Could not load image {image_path}.")
                break 
            
            frame = cv2.imread(os.path.join(image_path))
            image_name = image_path.strip().split('/')[-1]
            darknet_width = darknet.network_width(self.model)
            darknet_height = darknet.network_height(self.model)
            frame_resized = cv2.resize(frame, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
            img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
            darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
                    
            dets = darknet.detect_image(self.model, self.names, img_for_detect, thresh=opt.thresh)
            detections_adjusted = []
            for label, conf, bbox in dets:
                bbox_adjusted = self.convert_to_original(frame, darknet_height, darknet_width, bbox)
                _, _, box_width, box_height = bbox_adjusted
                if label in ['cat', 'dog'] and (box_width > 48 and box_width < 128) or (box_height > 48 and box_height < 128):
                    detections_adjusted.append((str(label), conf, bbox_adjusted))
                    frame = darknet.draw_boxes(detections_adjusted, frame, self.colors)
                    cv2.imwrite(os.path.join(self.videos_dir, f'{image_name}'), frame)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--source_url", type=str, default="", help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--input_type", type=str, default='video', help='type of the input source')
    parser.add_argument("--out_filename", type=str, default="", help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights", help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true', help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true', help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg", help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data", help="path to data file")
    parser.add_argument("--thresh", type=float, default=.6, help="remove detections with confidence below this value")
    opt = parser.parse_args()
    
    print(opt)
    engine = Engine(opt)
    if opt.input_type == 'video':
        engine.detect(opt)
    elif opt.input_type == 'image':
        engine.detect_frame(opt) 
    else:
        print('[Error] You must set input type as video or image')