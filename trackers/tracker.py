from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import numpy as np
sys.path.append('../')
from utils import get_centre_of_bbox, get_bbox_width 
import cv2

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):

        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks={
            'players':[], 
            'referee':[],
            'ball':[]        
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            #Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert Goalkeeper to Player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"] 

            #Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks['players'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox" : bbox} 

                if cls_id == cls_names_inv['referee']:
                    tracks["referee"][frame_num][track_id] = {'bbox' : bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {'bbox' : bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_elipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_centre, _ = get_centre_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
                frame,
                center=(x_centre,y2),
                axes=(int(0.9*width), int(0.28*width)),
                angle=0.0,
                startAngle=-25,
                endAngle=205,
                color = color,
                thickness=3,
                lineType=cv2.LINE_AA
        )

        cv2.ellipse(
                frame,
                center=(x_centre,y2),
                axes=(int(0.55*width), int(0.17*width)),
                angle=0.0,
                startAngle=-25,
                endAngle=205,
                color = color,
                thickness=1,
                lineType=cv2.LINE_AA
        )

        rectangle_width = 38
        rectangle_height = 18
        x1_rect = int(x_centre - rectangle_width//2)
        x2_rect = int(x_centre + rectangle_width//2)
        y1_rect = int((y2 - rectangle_height//2) + 18)
        y2_rect = int((y2 + rectangle_height//2) + 18)

        if track_id is not None:
            radius = rectangle_height//2
            y_mid = (y1_rect + y2_rect)//2

            cv2.rectangle(frame, (x1_rect+radius, y1_rect), (x2_rect-radius, y2_rect), color, cv2.FILLED)
            cv2.circle(frame, (x1_rect+radius, y_mid), radius, color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (x2_rect-radius, y_mid), radius, color, cv2.FILLED, cv2.LINE_AA)

            label = f"{track_id}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)

            cv2.putText(frame,
                        label,
                        (x_centre - text_w//2, y_mid + text_h//2),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.5,
                        (255,255,255),
                        1,
                        cv2.LINE_AA
            )


        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_centre_of_bbox(bbox)

        marker_points = np.array([
            [x, y-6],
            [x-14, y-24],
            [x+14, y-24]
        ])
        cv2.drawContours(frame, [marker_points], 0, (0,0,0), 4, cv2.LINE_AA)
        cv2.drawContours(frame, [marker_points], 0, color, cv2.FILLED, cv2.LINE_AA)
        cv2.drawContours(frame, [marker_points], 0, (255,255,255), 1, cv2.LINE_AA)

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames=[]
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]
        
            for track_id, player in player_dict.items():
                frame = self.draw_elipse(frame, player["bbox"],(0,0,255), track_id)
                
            for track_id, referee in referee_dict.items():
                frame = self.draw_elipse(frame, referee["bbox"],(0,190,255), track_id)

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (255,220,0))
            
                
            output_video_frames.append(frame)

        return output_video_frames