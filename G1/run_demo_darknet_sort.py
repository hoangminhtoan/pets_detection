from ctypes import *
import os
import cv2
#import darknet
import argparse
from datetime import datetime
from tqdm import tqdm
import glob
import numpy as np
import torch 
import torchvision
import torch.nn as nn

from configs import config
from utils import *
from tool.sort import *

dt_string = datetime.now().strftime("%Y%m%d")

def apply_classifier(x, model, img, im0):
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
        folder_name = opt.source_url.strip().split('/')[-2].replace('\n', '')
            
        if not os.path.exists(os.path.join(config.RESULT_DIR, f'{dt_string}_videos', f'{folder_name}', 'frames')):
            os.makedirs(os.path.join(config.RESULT_DIR, f'{dt_string}_videos', f'{folder_name}', 'frames'))
        
        
        self.videos_dir = os.path.join(config.RESULT_DIR, f'{dt_string}_videos', f'{folder_name}')
        self.frames_dir = os.path.join(config.RESULT_DIR, f'{dt_string}_videos', f'{folder_name}', 'frames')
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

        # create tracker
        self.tracker = Sort()
        memory = {}

        #load video
        paths = list(glob.glob(os.path.join(opt.source_url, "*/*.mp4")))
        paths.sort()
        if not os.path.isfile(opt.source_url):
            print(f"Could not load video {opt.source_url}.")
            return
        
        cap = cv2.VideoCapture(opt.source_url)
        
        # Set resolution
        fps = cap.get(cv2.CAP_PROP_FPS)
        
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
        FLAG_SAVE = False
        
        while cap.isOpened():
            frame_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_step == 0:
                self.duration += 1
                im0 = cv2.resize(frame, (video_width,video_height))
                frame_resized = cv2.resize(frame, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
                img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
                darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
                
                dets = darknet.detect_image(self.model, self.names, img_for_detect, thresh=opt.thresh)

                detections_adjusted = []
                if len(dets):   
                    for label, conf, bbox in dets:
                        bbox_adjusted = self.convert_to_original(frame, darknet_height, darknet_width, bbox)
                        x0, y0, box_width, box_height = bbox_adjusted
                        if label in ['cat', 'dog'] and ((box_width > 48 and box_width < 128) or (box_height > 48 and box_height < 128)):
                            #detections_adjusted.append((str(label), conf, bbox_adjusted))
                            detections_adjusted.append([x0, y0, x0 + box_width, y0 + box_height, float(conf) / 100])
                            #FLAG_SAVE = True
                            #frame = darknet.draw_boxes(detections_adjusted, frame, self.colors)
           
                    #if FLAG_SAVE:
                    #    FLAG_SAVE = False
                    #    images_list.append(frame)

                dets = np.asarray(detections_adjusted)
                tracks = self.tracker.update(dets)
                boxes = []
                indexIDs = []
                c = []
                previous = memory.copy()
                memory = {}

                for track in tracks:
                    boxes.append([track[0], track[1], track[2], track[3]])
                    indexIDs.append(int(track[4]))
                    memory[indexIDs[-1]] = boxes[-1]
                
                if len(boxes) > 0:
                    i = int(0)
                    for box in boxes:
                        x0, y0, box_width, box_height = int(box[0]), int(box[1]), int(box[2]), int(box[3])
			            cv2.rectangle(frame, (x, y), (w, h), self.color, 2)

                        if indexIDs[i] in previous:
                            previous_box = previous[indexIDs[i]]
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                            p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                            #cv2.line(frame, p0, p1, color, 3)

                            if intersect(p0, p1, line[0], line[1]):
                                counter += 1

                        text = "{}: {:.4f}".format(self.names[classIDs[i]], confidences[i])
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        i += 1
            '''
            if self.duration == self.fps_target:
                if len(images_list) > 2:
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print("[Warning!] at time {} Pet(s) were detected!".format(now))
                    if not os.path.exists(os.path.join(self.frames_dir, video_name)):
                        os.makedirs(os.path.join(self.frames_dir, video_name))           
                    for idx, image in enumerate(images_list):
                        prefix = '{:07}.jpg'
                        cv2.imwrite(os.path.join(self.frames_dir, video_name, f'{offset}_{now}_{video_name}_{prefix.format(idx)}'), image)
                self.duration = 0
                images_list = []
                offset+=1
            '''
                

            
            # Write Video
            out.write(frame)
            
            # Show prediction
            # cv2.imshow(WINDOW_NAME, frame)
            
            #if cv2.waitKey(24) & 0xFF == ord('q')
            #   break
        cap.release()
        # cv2.destroyAllWindows()

    def detect_from_frame(self, opt):
        import darknet
        #load detector model
        self.model, self.names, self.colors = darknet.load_network(opt.config_file, opt.data_file, opt.weights, batch_size=1)
        darknet_width = darknet.network_width(self.model)
        darknet_height = darknet.network_height(self.model)
        
        extensions = ['jpg', 'JPG', 'png', 'PNG']
        for _, image_file in tqdm(enumerate(os.listdir(opt.source_url)), total=len(os.listdir(opt.source_url))):
            ext = image_file[-3:]
            if ext in extensions:
                frame = cv2.imread(os.path.join(opt.source_url, image_file))
                frame_resized = cv2.resize(frame, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
                img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
                darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
                
                dets = darknet.detect_image(self.model, self.names, img_for_detect, thresh=opt.thresh)
            
                if len(dets) == 0:
                    print("No pet(s) detected!")
                else:    
                    print("Pet(s) detection!")
                    detections_adjusted = []
                    for label, conf, bbox in dets:
                        bbox_adjusted = self.convert_to_original(frame, darknet_height, darknet_width, bbox)
                        _, _, box_width, box_height = bbox_adjusted
                        if label in ['cat', 'dog'] and ((box_width > 48 and box_width < 128) or (box_height > 48 and box_height < 128)):
                            detections_adjusted.append((str(label), conf, bbox_adjusted))
                            FLAG_SAVE = True
                            frame = darknet.draw_boxes(detections_adjusted, frame, self.colors)
                            cv2.imwrite(os.path.join(self.frames_dir, f'{image_file}'), frame)

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
        engine.detect_from_frame(opt)
    else:
        print('[Error] You must set input type as video or image')
    
    
