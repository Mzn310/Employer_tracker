import cv2
import numpy as np
import time
import os
from datetime import datetime
import urllib.request
import threading
import queue
import tempfile
from werkzeug.utils import secure_filename


class EmployeeTracker:
    def __init__(self):
        self.is_running=False
        self.tracking_thread = None
        self.frame_queue=queue.Queue()
        self.current_frame=None
        self.lock=threading.Lock()

        self.employee_present=False
        self.absence_start_time=None
        self.absence_logged=False
        self.last_present_time=None
        self.frames_processed=0

        # config
        self.camera_source=0
        self.absence_threshold=10
        self.confidence_threshold=0.5
        self.monitor_area=None
        self.save_intervale=20
        self.output_dir="output"
        self.source_type="webcam"
        self.uploaded_video_path=None

        self.model_dir="yolo_model"
        self.net=None
        self.ouput_layers=None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists("logs"):
            os.makedirs("logs")

        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        self.log_file_path=os.path.join("logs", "employee_log.txt")
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w") as f:
                f.write("Employee Tracking Log\n")
                

    def setup_model(self):
        if not self._download_yolo_files():
            return False

        weights_path=os.path.join(self.model_dir,"yolov4-tiny.weights")
        config_path=os.path.join(self.model_dir,"yolov4-tiny.cfg")

        try:
            self.net=cv2.dnn.readNetFromDarknet(config_path, weights_path)

            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Get output layer names
            layer_names=self.net.getLayerNames()
            try:
                self.ouput_layers=[layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
            except:
                self.ouput_layers=[layer_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]


            return True
        except Exception as e:
            self.log_event(f"Error loading YOLO model: {e}")
            return False

    def _download_yolo_files(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        files = {
            "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
            "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
            "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
        }
        
        for filename, url in files.items():
            filepath = os.path.join(self.model_dir, filename)
            if not os.path.exists(filepath):
                try:
                    urllib.request.urlretrieve(url, filepath)
                    self.log_event(f"Downloaded {filename} successfully")
                except Exception as e:
                    self.log_event(f"Error downloading {filename}: {e}")
                    return False
        
        return True

    def log_event(self, message):
        with open(self.log_file_path,"a") as log_file:
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] {message}\n")


    def get_logs(self,max_lines=100):
        if not os.path.exists(self.log_file_path):
            return []

        try:
            with open(self.log_file_path,"r") as f:
                lines=f.readlines()
                return lines[-max_lines:] if lines else []
        except:
            return ["Error reading log file"]

    def get_status(self):
        with self.lock:
            status={
                "status":"active" if self.is_running else "inactive",
                "employee_present":self.employee_present,
                "frames_processed":self.frames_processed,
                }

            if not self.employee_present and self.absence_start_time is not None:
                status["absence_duration"]=time.time()-self.absence_start_time
            else:
                status["absence_duration"]=0

            return status

    def upload_video(self,video_file,config):
        if self.is_running:
            return {"status":"error", "message":"Tracking is already running"}

        try:
            filename=secure_filename(video_file.filename)
            file_path=os.path.join("uploads", f"{int(time.time())}_{filename}")
            video_file.save(file_path)

            self.log_event(f"Video uploaded: {filename}")
            self.upload_video_path=file_path
            self.source_type="upload"

            return self.start_tracking(config)

        except Exception as e:
            self.log_event(f"Error uploading video: {e}")
            return {"status":"error", "message":str(e)}


    def start_tracking(self,config):
        if self.is_running:
            return {"status":"error", "message":"Tracking is already running"}

        self.source_type=config.get("source_type","webcam")

        if self.source_type=="upload" and self.uploaded_video_path is None:
            return {"status":"error", "message":"No video uploaded available"}
        elif self.source_type == "webcam":
            self.camera_source = config.get("camera_source", 0)
            if isinstance(self.camera_source, str) and self.camera_source.isdigit():
                self.camera_source = int(self.camera_source)
        elif self.source_type == "custom":
            self.camera_source = config.get("camera_source", "")
        elif self.source_type in ["gdrive", "s3"]:
            return {"status": "error", "message": f"{self.source_type} source not yet implemented"}

        self.absence_threshold = float(config.get("absence_threshold", 5))
        self.confidence_threshold = float(config.get("confidence", 0.5))
        
        # Setup area method
        area_method = config.get("area_method", "auto")
        
        # Make sure the model is set up
        if self.net is None:
            if not self.setup_model():
                return {"status": "error", "message": "Failed to set up detection model"}
        
        # Open camera to get frame dimensions
        cap = self._open_camera()
        if cap is None:
            return {"status": "error", "message": "Failed to open video source"}
            
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"status": "error", "message": "Failed to read initial frame"}
            
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (600, int(frame.shape[0] * 600 / frame.shape[1])))
        height, width = frame.shape[:2]

        if area_method == "manual":
            # Parse manually specified area
            try:
                coords = config.get("manual_coords", "0.1,0.1,0.9,0.9")
                x1, y1, x2, y2 = map(float, coords.split(','))
                self.monitor_area = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )
                self.log_event(f"Using manually specified area: {self.monitor_area}")
            except:
                # Default if parsing fails
                self.monitor_area = (int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9))
                self.log_event("Failed to parse manual coords, using default area")
        else:
            # Auto-detect desk area
            self.monitor_area = self._detect_desk_area(cap)
            self.log_event(f"Auto-detected desk area: {self.monitor_area}")

         # Release initial camera
        cap.release()
        
        # Reset tracking variables
        self.employee_present = False
        self.absence_start_time = None
        self.absence_logged = False
        self.last_present_time = time.time()
        self.frames_processed = 0
        
        # Log system start
        self.log_event(f"Tracking started using {self.source_type} source")
        
        # Start tracking thread
        self.is_running = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        return {"status": "success", "message": "Tracking started"}


    