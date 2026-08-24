import cv2
import numpy as np
import time
import os
from datetime import datetime
import urllib.request
import threading
import queue
import tempfile
from sympy import re
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
        self.output_layers=None

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
                self.output_layers=[layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
            except:
                self.output_layers=[layer_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]


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


    def stop_tracking(self):
        if not self.is_running:
            return {"status": "error", "message": "Tracking is not running"}

        self.is_running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)

        self.log_event("Tracking system stopped")

        if self.source_type == "upload" and self.uploaded_video_path:
            try:
                os.remove(self.uploaded_video_path)
                self.log_event(f"Deleted uploaded video: {self.uploaded_video_path}")
                self.uploaded_video_path = None
            except:
                self.log_event(f"Failed to delete uploaded video: {self.uploaded_video_path}")
        return {"status": "success", "message": "Tracking stopped"}

    def get_current_frame(self):
        """Get the latest processed frame for the video feed"""
        with self.lock:
            if self.current_frame is not None:
                ret,jpeg=cv2.imencode(".jpg",self.current_frame)
            if ret:
                return jpeg.tobytes()

        # Return a blank frame if no current frame is available
        blank=np.zeros((300,400,3),dtype=np.uint8)
        blank[:]=[50,50,50] # Dark gray background
        cv2.putText(blank,"Loading...",(120,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        ret,jpeg=cv2.imencode(".jpg",blank)
        return jpeg.tobytes()

    def open_camera(self):
        try:
            if self.source_type == "upload" and self.uploaded_video_path:
                # Use uploaded video file 
                cap=cv2.VideoCapture(self.uploaded_video_path)
                if not cap.isOpened():
                   self.log_event(f"Failed to open uploaded video: {self.uploaded_video_path}")
                   return None
                return cap
            elif self.source_type == "webcam":
                if isinstance(self.camera_source, int) or self.camera_source.isdigit():
                    cap=cv2.VideoCapture(int(self.camera_source)) 
                else:
                    cap=cv2.VideoCapture(self.camera_source)
            elif self.source_type == "custom":
                cap=cv2.VideoCapture(self.camera_source)
            else:
                self.log_event(f"Unsupported source type: {self.source_type}")
                return None

            if not cap.isOpened():
                self.log_event(f"Failed to open camera source: {self.camera_source}")
                return None

            return cap
        except Exception as e:
            self.log_event(f"Error opening camera: {e}")
            return None 

    def detect_desk_area(self, cap):
        self.log_event("Starting desk area detection...")

        desk_candidates = []
        frame_count=0
        max_frames=5

        while frame_count < max_frames:
            ret,frame=cap.read()
            if not ret:
                break

            frame=cv2.resize(frame,(600,int(frame.shape[0]*600/frame.shape[1])))
            height,width=frame.shape[:2]

            blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
            self.net.setInput(blob)

            try:
                detections=self.net.forward(self.output_layers)
            except Exception as e:
                self.log_event(f"Error during detection: {e}")
                break

            desks=[]

            for detection in detections:
                for obj in detection:
                    scores=obj[5:]
                    class_id=np.argmax(scores)
                    confidence=scores[class_id]

                     # COCO class IDs: 0=person, 56=chair, 60=dining table, 62=tv, 63=laptop, 64=mouse, 65=keyboard, 73=book
                    # We're looking for furniture objects that indicate a desk area
                    desk_related_classes = [56, 60, 63, 64, 65]

                    if confidence > self.confidence_threshold and class_id in desk_related_classes:
                        center_x=int(obj[0]*width)
                        center_y=int(obj[1]*height)
                        w=int(obj[2]*width)
                        h=int(obj[3]*height)

                        x=int(center_x - w/2)
                        y=int(center_y - h/2)
                        desks.append((x, y, w, h,confidence,class_id))

            if desks:
                desk_candidates.extend(desks)

            frame_count+=1
            if self.source_type == "upload":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # If we don't have enough desk candidates, use a default area
        if len(desk_candidates) < 2:
            self.log_event("Not enough desk objects detected. Using default desk area.")
            # Default to middle-bottom half of the frame
            height, width = frame.shape[:2]
            desk_area = [int(width * 0.2), int(height * 0.4), int(width * 0.8), int(height * 0.9)]
            return tuple(desk_area)
        
        # Group desk objects to determine the desk area
        x_coords = [d[0] for d in desk_candidates]
        y_coords = [d[1] for d in desk_candidates]
        max_x = max([d[0] + d[2] for d in desk_candidates])
        max_y = max([d[1] + d[3] for d in desk_candidates])
        
        # Create a bounding box that covers all desk-related objects
        # with some padding (10% on each side)
        min_x = max(0, int(min(x_coords) - 0.1 * width))
        min_y = max(0, int(min(y_coords) - 0.1 * height))
        max_x = min(width, int(max_x + 0.1 * width))
        max_y = min(height, int(max_y + 0.1 * height))
        
        desk_area = (min_x, min_y, max_x, max_y)

        return desk_area

    def tracking_loop(self):
        cap=self.open_camera()
        if cap is None:
            self.log_event("Failed to open camera in tracking loop")
            self.is_running=False
            return
        try:
            last_save_time=time.time()

            video_ended=False

            while self.is_running:
                ret,frame=cap.read()

                if not ret:
                    if self.source_type=="upload":
                        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                        ret,frame=cap.read()
                        if ret:
                            if not video_ended:
                                self.log_event("End of video reached")
                                video_ended=True
                        else:
                            self.log_event("Failed to loop video")
                            break
                    else:
                        self.log_event("Failed to read frame")
                        break

                frame=cv2.resize(frame,(600,int(frame.shape[0]*600/frame.shape[1])))
                height,width=frame.shape[:2]

                proceessed_frame, employee_detected=self.process_frame(frame)

                current_time=time.time()

                if employee_detected:
                    if not self.employee_present:
                        self.employee_present=True
                        if self.absence_start_time is not None:
                            absence_duration=current_time-self.absence_start_time
                            self.log_event(f"Employee returned after {absence_duration:.1f} seconds")
                            self.absence_start_time=None
                            self.absence_logged=False

                    self.last_present_time=current_time
                else:
                    if self.employee_present:
                        self.employee_present=False
                        self.absence_start_time=current_time
                    elif self.absence_start_time is not None:
                        absence_duration=current_time-self.absence_start_time
                        if absence_duration>= self.absence_threshold and not self.absence_logged:
                            self.log_event("Employee absence detected")
                            self.absence_logged=True


                with self.lock:
                    self.current_frame=proceessed_frame
                    self.frames_processed+=1

                if current_time-last_save_time>10:
                    frame_filename=os.path.join(self.output_dir,f"frame_{self.frames_processed:06d}.jpg")
                    cv2.imwrite(frame_filename,proceessed_frame)
                    last_save_time=current_time


                if self.source_type=="upload":
                    time.sleep(0.03) # ~30 fps
                else:
                    time.sleep(0.01)
        except Exception as e:
            self.log_event(f"Error in tracking loop: {str(e)}")
        finally:
            cap.release()
            self.is_running=False
            self.log_event("Tracking loop ended")

    def process_frame(self,frame):
        height,width=frame.shape[:2]
        employee_detected=False

        blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
        self.net.setInput(blob)

        try:
            detections=self.net.forward(self.output_layers)
        except Exception as e:
            self.log_event(f"Error during detection: {e}")
            return frame, False

        boxes=[]
        confidences=[]
        class_ids=[]

        for detection in detections:
            for obj_detection in detection:
                scores=obj_detection[5:]
                class_id=np.argmax(scores)
                confidence=scores[class_id]

                if confidence>self.confidence_threshold and class_id==0: 
                    center_x=int(obj_detection[0]*width)
                    center_y=int(obj_detection[1]*height)
                    w=int(obj_detection[2]*width)
                    h=int(obj_detection[3]*height)

                    x=int(center_x-w/2)
                    y=int(center_y-h/2)

                    boxes.append([x,y,w,h])
                    confidence.append(float(confidence))
                    class_ids.append(class_id)
        if len(boxes)>0:
            indices=cv2.dnn.NMSBoxes(boxes,confidence,self.confidence_threshold,0.4)

            if len(indices)>0:
                for i in indices.flatten():
                    x,y,w,h=boxes[i]

                    person_box=(x,y,x+w,y+h)

                    x_intersection=max(self.monitor_area[0],person_box[0])
                    y_intersection=max(self.monitor_area[1],person_box[1])
                    w_intersection=min(self.monitor_area[2],person_box[2])-x_intersection
                    h_intersection=min(self.monitor_area{3},person_box[3])-y_intersection

                    is_in_desk_area=False
                    if w_intersection>0 and h_intersection>0:
                        intersection_area=w_intersection*h_intersection
                        person_area=w*h
                        overlap_ratio=intersection_area/person_area

                        if overlap_ratio>0.3:
                            employee_detected=True
                            is_in_desk_area=True

                    if is_in_desk_area:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                        label=f"Employe: {confidences[i]:.2f}"
                        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                    else:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                        label=f"Person:{confidences[i]:.2f}"
                        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            cv2.rectangle(frame,(self.monitor_area[0],self.monitor_area[1]),(self.monitor_area[2],self.monitor_area[3]),(0,255,0),2)

            status_text= "Status: PRESENT" if employee_detected else "Status: ABSENT"
            cv2.putText(frame, status_text, (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0) if employee_detected else (0,0,255),2)

            if not employee_detected and self.absence_start_time is not None:
                absence_duration=time.time()-self.absence_start_time
                duration_text=f"Absence: {absence_duration:.1f}s"
                cv2.putText(frame,duration_text,(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame,f"Time: {timestamp}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(frame,f"Source: {self.source_type}",(10,110),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

            return frame ,employee_detected
