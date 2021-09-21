import argparse
import time
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, voc_to_yolo
from utils.datasets import letterbox, letterbox2
from utils.plots import plot_one_box
from utils.configs import config

#dt_string = datetime.now().strftime("%Y%m%d")
WINDOW_NAME = 'Pets Detection'


class Engine():
    def __init__(self, opt) -> None:
        self.fps_target = 10 # target 10 FPS
        self.frame_step = 1 # select every frame
        self.duration = 0
        self.submodel_img_size = 128
        
        # Create folder to save result videos
        folder_name = opt.source_url.strip().split('/')[-2].replace('\n', '')
            
        if not os.path.exists(os.path.join(config.RESULT_DIR, f'false_alarm_videos', f'{folder_name}')):
            os.makedirs(os.path.join(config.RESULT_DIR, f'false_alarm_videos', f'{folder_name}'))
        
        
        self.folder_dir = os.path.join(config.RESULT_DIR, f'false_alarm_videos', f'{folder_name}')
    
    def load_model(self, opt, img_size=None):
        # speed up constant image size inference
        cudnn.benchmark = True
        #load model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = attempt_load(opt.weights, map_location=self.device)
        
        self.stride = int(model.stride.max())  # model stride
        # Get names and colors
        self.names = model.module.names if hasattr(model, 'module') else model.names
        #self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.colors = [[255, 153, 51], [255, 0, 255], [0, 102, 204]] #cat, dog
        
        if self.device.type != 'cpu':
            if img_size is None:
                model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(self.device).type_as(next(model.parameters())))  # run once
            else:
                model(torch.zeros(1, 3, img_size, img_size).to(self.device).type_as(next(model.parameters())))  # run once
            
        return model
        
    def detect(self, opt):
        # load model
        model = self.load_model(opt, img_size=None)
        #sub_model = self.load_model(opt, img_size=self.submodel_img_size)
        
        if opt.half:
            model = model.half()
            #sub_model = sub_model.half()

        #load video
        if not os.path.isfile(opt.source_url):
            print(f"Could not load video {opt.source_url}. Check directory again!")
            return
        
        cap = cv2.VideoCapture(opt.source_url)

        # set resolution
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps >= self.fps_target:
            self.frame_step = round(fps / self.fps_target) # if fps = 20 skips 2 frames
        print(f'input video {opt.source_url} with frame rate: {fps} => skip {self.frame_step} frame(s)')
            
        w = 1920 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 1920
        h = 1080 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 1080
        
        # set output video
        video_name = opt.source_url.strip().split('/')[-1].replace('\n', '')
        #out = cv2.VideoWriter(os.path.join(self.videos_dir, f'{dt_string}_{video_name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        frame_idx = -1
        box_idx = 0
        offset = 0
        images_list = []
        orig_images_list = []
        bboxes_list = defaultdict(list)
        FLAG_SAVE = False
        
        while cap.isOpened():
            frame_idx += 1
        
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (w, h))
            tmp_frame = frame.copy()
            if frame_idx % self.frame_step == 0:
                self.duration += 1
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
                preds = non_max_suppression(preds, opt.conf_thres,  opt.iou_thres, classes=opt.classes)
                
                # Process detections
                for i, det in enumerate(preds):
                    if len(det):
                    # Rescale boxes from img to original size
                        orig_det = det
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                            bbox_width = int(xyxy[2]) - int(xyxy[0])
                            bbox_height = int(xyxy[3]) - int(xyxy[1])
                            label = f'{self.names[int(cls)]} {conf*100:.1f}%'
                            #if self.reaffirm_pets(submodel, bbox, video_name, tmp_frame, opt):
                            if bbox_height < 198 and bbox_width < 198:
                                FLAG_SAVE = True
                                bboxes_list[box_idx].append(voc_to_yolo(4, bbox, frame)) # cat: 0, dog: 1, head: 2, human: 3, others: 4
                                plot_one_box(xyxy, tmp_frame, label=label, color=self.colors[0], line_thickness=3)
                            
                if FLAG_SAVE:
                    FLAG_SAVE = False
                    images_list.append(tmp_frame)
                    orig_images_list.append(frame)
                    box_idx += 1

            if self.duration == self.fps_target:               
                if len(images_list) > 0: 
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print("[Warning!] at time {} Pet(s) were detected!".format(now))
                    if not os.path.exists(os.path.join(self.folder_dir, video_name)):
                        os.makedirs(os.path.join(self.folder_dir, video_name, 'images'))
                        os.makedirs(os.path.join(self.folder_dir, video_name, 'labels'))
                        
                    for idx, image in enumerate(images_list):
                        prefix_img = '{:07}.jpg'
                        prefix_txt = '{:07}.txt'
                        cv2.imwrite(os.path.join(self.folder_dir, video_name, f'{offset}_{video_name}_{prefix_img.format(idx)}'), image)
                        cv2.imwrite(os.path.join(self.folder_dir, video_name, 'images', f'{offset}_{video_name}_{prefix_img.format(idx)}'), orig_images_list[idx])
                        # save bounding box
                        for box in bboxes_list[idx]:
                            with open(os.path.join(self.folder_dir, video_name, 'labels', f'{offset}_{video_name}_{prefix_txt.format(idx)}'), 'a') as f:
                                f.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n')
                                
                self.duration = 0
                images_list = []
                orig_images_list = []
                bboxes_list = defaultdict(list)
                box_idx = 0
                offset += 1
        
        cap.release()

    def reaffirm_pets(self, model, bbox, video_name, frame, opt):
        x0, y0, x1, y1 = bbox
        if x1 - x0 < self.img_size_for_sub_model:
            w = self.img_size_for_sub_model
        else:
            w = max(x1 - x0, y1 - y0) + 5
        if y1 - y0 < self.img_size_for_sub_model:
            h = self.img_size_for_sub_model
        else:
            h = max(x1 - x0, y1 - y0) + 5
            
        x0, y0 = max(x0 - w // 2, 0), max(y0 - h // 2, 0)
        x1, y1 = min(x1 + w // 2, frame.shape[1] - 1), min(y1 + h // 2, frame.shape[0] - 1)

        # Resize bounding box to size (128, 128)
        img = cv2.resize(frame[y0 : y1, x0 : x1], (self.img_size_for_sub_model, self.img_size_for_sub_model))

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
                return True 
            
        return False
        
    def detect_from_frame(self, opt):
        model = self.load_model(opt) # load detection from main image (608)
        #submodel = self.load_model(opt, self.img_size_for_sub_model) # re-check bounding box (128)
        if opt.half:
            model = model.half()
            #submodel = submodel.half()
        
        extensions = ['jpg', 'JPG', 'png', 'PNG']
        for _, image_file in tqdm(enumerate(os.listdir(opt.source_url)), total=len(os.listdir(opt.source_url))):
            ext = image_file[-3:]
            if ext in extensions:
                frame = cv2.imread(os.path.join(opt.source_url, image_file))
                frame = cv2.resize(frame, (1920, 1080))
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
                
                if len(preds) == 0:
                    print("No pet(s) detected!")
                else:    
                    print("Pet(s) detection!")
                    for i, det in enumerate(preds):
                        if len(det):
                            # Rescale boxes from img to original size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                            
                            for *xyxy, conf, cls in reversed(det):
                                label = f'{self.names[int(cls)]} {conf*100:.1f}%'
                                bbox_width = int(xyxy[2]) - int(xyxy[0])
                                bbox_height = int(xyxy[3]) - int(xyxy[1])
                                if bbox_height < 160 and bbox_width < 160:
                                    #if self.reaffirm_pets(submodel, frame[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], opt):
                                    plot_one_box(xyxy, frame, label=label, color=self.colors[0], line_thickness=2)
                                    cv2.imwrite(os.path.join(self.frames_dir, f'{image_file}'), frame)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--source-url', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--input-type', type=str, default='video', help='type of the input source: video/image')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        engine = Engine(opt)
        if opt.input_type == 'video':
            engine.detect(opt)
        elif opt.input_type == 'image':
            engine.detect_from_frame(opt)
        else:
            print('[Error] You must set input type as video or image')