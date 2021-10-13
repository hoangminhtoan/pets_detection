from ctypes import *
import os
import cv2
import argparse
import glob
from datetime import datetime
import numpy as np
import torch 
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import collections
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, fbeta_score

from tool.config import config
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox


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
        # Load darknet model
        # speed up constant image size inference
        cudnn.benchmark = True

        #load detector model
        g1_model = Darknet(opt.config_file)
        g1_model.load_weights(opt.weight_g1)
        g1_model.to(self.device)
        g1_model.eval()

        self.names = load_class_names(opt.names_file) 
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Load cloud model detector
        g2_model = attempt_load(opt.weight_cloud, map_location=self.device)
        self.stride = int(g2_model.stride.max())  # model stride
        if self.device.type != 'cpu':
            g2_model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(self.device).type_as(next(g2_model.parameters())))  # run once

        return g1_model, g2_model
    
    def load_classifier(self, name='resnet18', num_classes=2):
        model = torchvision.models.__dict__[name](pretrained=True)
        print("Loading Classifier ....")
        # Reshape output to n classes
        filters = model.fc.weight.shape[1]
        model.fc.bias = nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        model.fc.weight = nn.Parameter(torch.zeros(num_classes, filters), requires_grad=True)
        model.fc.out_features = num_classes
        
        modelc = model.load_state_dict(torch.load(os.path.join('weights', f'{name}_224_cat_dog.pt'), map_location=self.device))
        modelc = model.to(self.device).eval()
        
        return modelc


    def detect(self, opt):
        #load detector model
        g1_model, g2_model = self.load_detector_model(opt)

        # load classifier model
        modelc = self.load_classifier()

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
                img = cv2.resize(frame, (g1_model.width, g1_model.height), interpolation=cv2.INTER_LINEAR) # resize input frame into input g1 network size
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
                preds = g1_model(img)

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
                        if (bbox[0] > 10 and bbox[0] < video_width - 10) and (bbox[1] > 10 and bbox[1] < video_height - 10):
                            #label = f'{self.names[int(cls_id)]} {conf*100:.1f}'
                            FLAG_SAVE = True
                            #frame = utils.plot_one_box(xyxy, frame, label=label, color=self.colors[0], line_thickness=2) #output of g1_model

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
                        #cv2.imwrite(os.path.join(self.frames_dir, video_name, f'{offset}_{now}_{video_name}_{prefix.format(d)}'), images_list[d])
                        frame = self.detect_frame(g2_model, opt, frame, frame_name=f'{offset}_{now}_{video_name}_{prefix.format(d)}') # call cloud model

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

    def detect_frame(self, model, opt, frame, img_name):
        # set image size
        img = letterbox(frame, opt.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if opt.half else img.float() 
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
                
        # detect object
        preds = model(img, augment=False)[0]
                
        # Apply NMS
        preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres, classes=opt.classes)
                
        for i, det in enumerate(preds):
            if len(det):
                # Rescale boxes from img to original size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                            
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf*100:.1f}%'
                    bbox_width = int(xyxy[2]) - int(xyxy[0])
                    bbox_height = int(xyxy[3]) - int(xyxy[1])
                    if bbox_height > 48 or bbox_width > 48:
                        plot_one_box(xyxy, frame, label=label, color=self.colors[0], line_thickness=2)
                        cv2.imwrite(os.path.join(self.frames_dir, f'{img_name}'), frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--source_url", type=str, default="", help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--input_type", type=str, default='video', help='type of the input source')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument("--weight_g1", default="dcm.weight", help="g1 weights path")
    parser.add_argument("--weight_cloud", default="dcm.pt", help="cloud weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg", help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data", help="path to data file")
    parser.add_argumnet("--names_file", default="./cfg/x.names", hept="path to class names file")
    parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
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