from ctypes import *
import os
import cv2
import argparse
import glob
from datetime import datetime
import numpy as np
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import collections
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, fbeta_score

from tool.config import config
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet


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
    
    def load_detector_model(self, opt):
        # speed up constant image size inference
        cudnn.benchmark = True

        #load detector model
        model = Darknet(opt.config_file)
        model.load_weights(opt.weight_file)
        model.to(self.device)
        model.eval()

        self.names = load_class_names(opt.names_file) 
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        return model

    def detect(self, opt):
        #load detector model
        model = self.load_detector_model(opt)

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
            
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #1920
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #1080
            
        # Set output video
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
                frame = cv2.resize(frame, (video_width, video_height))
                img = cv2.resize(frame, (model.width, model.height), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.to(self.device)
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
            
                # detect object
                preds = model(img)

                # Apply NMS
                preds = utils.post_processing(img, opt.conf_thres, opt.iou_thres, preds)[0] # [x0, y0, x1, y1, cls_conf, cls_id]
                
                bnd_boxes = []
                max_conf = -100.0
                for *xyxy, conf, cls_id in preds:
                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    label = self.name[int(cls_id)]
                    #if label in ['cat', 'dog', 'human']: # and ((box_height > 48 and box_height < 128) or (box_width > 48 and box_width)):
                    if label in ['cat', 'dog'] and box_height > 48 and box_width > 48:
                        max_conf = max(max_conf, float(conf))
                    # Checking duplicated bounding box in current frame with the previous frame
                    if len(previous_box_list) == 0: # initialize list of bounding b
                        previous_box_list.append(bbox, conf, cls_id)
                        bnd_boxes.append(bbox)
                    else:
                        for b in previous_box_list:
                            if utils.bbox_iou(b, bbox) > 0.7:
                                #print(b, box)
                                FLAG_DUPLICATE = True
                                break
                            FLAG_DUPLICATE = False
                        if FLAG_DUPLICATE is False: 
                            previous_box_list.append(bbox, conf, cls_id)
                            bnd_boxes.append(bbox)
                            if len(previous_box_list) > 5:
                                previous_box_list.pop(0)
                    # End checking duplicated bounding box in current frame with the priveious frame
                    FLAG_SAVE = True
                    print(previous_box_list, bbox)
                # The bounding box should not be too close to the edge of frame
                for bbox, conf, cls_id in previous_box_list:
                    if (bbox[0] > 10 and bbox[0] < video_width - 10) and (bbox[1] > 10 and bbox[1] < video_height - 10):
                        label = f'{self.names[int(cls_id)]} {conf*100:.1f}'
                        frame = utils.plot_one_box(xyxy, frame, label=label, color=self.colors[0], line_thickness=2)

                if FLAG_SAVE:
                    FLAG_SAVE = False
                    images_list.append(frame)
                    if next_offset == 0 or next_offset == offset: # ignore 10 offsets after current detected offset
                        next_offset = offset + 10

                    tmp_idx_conf[idx] = max_conf
                    idx += 1

            if self.duration == self.fps_target * 2: # duration là  20 frames 
                if (next_offset == offset + 10) and len(images_list) > 2:
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
        model = self.load_detector_model(opt)

        #load images
        image_paths =  list(glob.glob(os.path.join(opt.source_url, "*.jpg")))
        image_paths.sort()

        for idx, image_path in enumerate(image_paths):
            print(f'Image {idx+1}/{len(image_paths)}\n')
            if not os.path.isfile(image_path):
                print(f"Could not load image {image_path}.\n")
                break 
            image_name = image_path.strip().split('/')[-1]

            frame = cv2.imread(os.path.join(image_path))
            # set image size
            img_width = frame.shape[1] # 1920
            img_height = frame.shape[0] #1080
            frame = cv2.resize(frame, (img_width, img_height))
            img = cv2.resize(frame, (model.width, model.height), interpolation=cv2.INTER_LINEAR)
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.to(self.device)
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # detect object
            preds = model(img)

            # Apply NMS
            preds = utils.post_processing(img, opt.conf_thres, opt.iou_thres, preds)[0] # [x0, y0, x1, y1, cls_conf, cls_id]

            for *xyxy, conf, cls_id in preds:
                xyxy[0] = int(xyxy[0]) * img_width
                xyxy[1] = int(xyxy[1]) * img_height
                xyxy[2] = int(xyxy[2]) * img_width
                xyxy[3] = int(xyxy[3]) * img_height
                box_width = xyxy[2] - xyxy[0]
                box_height = xyxy[3] - xyxy[1]
                label = self.names[int(cls_id)]

                if label in ['cat', 'dog'] and box_width > 48 and box_height > 48:
                    label = f'{label} {conf*100:.1f}'
                    frame = utils.plot_one_box(xyxy, frame, label=label, color=self.colors[0], line_thickness=2)
                    cv2.imwrite(os.path.join(self.videos_dir, f'{image_name}'), frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--source_url", type=str, default="", help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--input_type", type=str, default='video', help='type of the input source')
    parser.add_argument("--out_filename", type=str, default="", help="inference video name. Not saved if empty")
    parser.add_argument("--weight_file", default="yolov4.weights", help="yolo weights path")
    parser.add_argument("--ext_output", action='store_true', help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg", help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data", help="path to data file")
    parser.add_argumnet("--names_file", default="./cfg/x.names", hept="path to class names file")
    parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    opt = parser.parse_args()
    
    print(opt)
    with torch.no_grad():
        engine = Engine(opt)
        if opt.input_type == 'video':
            engine.detect(opt)
        elif opt.input_type == 'image':
            engine.detect_frame(opt) 
        else:
            print('[Error] You must set input type as video or image')