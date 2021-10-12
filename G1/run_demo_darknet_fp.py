from ctypes import *
import os
import cv2
#import darknet
import argparse
import glob
from datetime import datetime
import numpy as np
import torch 
import torchvision
import torch.nn as nn
import collections

from configs import config
from utils import xywh2xyxy, xyxy2xywh, scale_coords

dt_string = datetime.now().strftime("%Y%m%d")

def apply_classifier(x, model, darknet_height, darknet_width, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x

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
        
    def convert_to_original(self, image, darknet_height, darknet_width, bbox):
        x, y, w, h = bbox
        x = x / darknet_width
        y = y / darknet_height
        w = w / darknet_width
        h = h / darknet_height

        image_h, image_w, __ = image.shape

        orig_x       = int(x * image_w)
        orig_y       = int(y * image_h)
        orig_width   = int(w * image_w)
        orig_height  = int(h * image_h)

        bbox_converted = (orig_x, orig_y, orig_width, orig_height)

        return bbox_converted
    

    def detect(self, opt):
        import darknet
        #load detector model
        self.model, self.names, self.colors = darknet.load_network(opt.config_file, opt.data_file, opt.weights, batch_size=1)

        #load video
        #video_paths =  list(glob.glob(os.path.join(opt.source_url, "*.mp4")))
        #video_paths.sort()
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
        #out = cv2.VideoWriter(os.path.join(self.videos_dir, f'{dt_string}_{video_name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        frame_idx = -1
        offset = 0
        images_list = []
        tmp_idx_conf = {}
        idx = 0
        FLAG_SAVE = False
            
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
                    _, _, box_width, box_height = bbox_adjusted
                    if label in ['cat', 'dog', 'human']: # and ((box_height > 48 and box_height < 128) or (box_width > 48 and box_width)):
                        max_conf = max(max_conf, float(conf))
                        detections_adjusted.append((str(label), conf, bbox_adjusted))
                        FLAG_SAVE = True
                        frame = darknet.draw_boxes(detections_adjusted, frame, self.colors)
                    
                if FLAG_SAVE:
                    FLAG_SAVE = False
                    images_list.append(frame)
                    tmp_idx_conf[idx] = max_conf 
                    idx += 1

                
            if self.duration == self.fps_target * 2:
                if len(images_list) > 3:
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print("[Warning!] at time {} Pet(s) were detected!".format(now))
                    if not os.path.exists(os.path.join(self.frames_dir, video_name)):
                        os.makedirs(os.path.join(self.frames_dir, video_name)) 
                    sorted_tm_idx = dict(sorted(tmp_idx_conf.items(), key=lambda item: item[1], reverse=True))      

                    counter_idx = 0
                    for i, _ in sorted_tm_idx.items():
                        if counter_idx < 3:
                            prefix = '{:07}.jpg'
                            cv2.imwrite(os.path.join(self.frames_dir, video_name, f'{offset}_{now}_{video_name}_{prefix.format(i)}'), images_list[i])
                        counter_idx += 1
                self.duration = 0
                images_list = []
                tmp_idx_conf = {}
                offset+=1
                idx = 0
                
                # Write Video
                # out.write(frame)
                
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
    
    