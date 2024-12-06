import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time
import multiprocessing

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / float(area1 + area2 - intersection)
    return iou

def get_box_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

def is_moving_towards_camera(prev_center, current_center, frame_height):
    return current_center[1] > prev_center[1]

def calculate_speed(prev_center, current_center, fps, pixels_per_meter):
    # Calculate distance moved in pixels
    distance_pixels = np.sqrt((current_center[0] - prev_center[0])**2 + (current_center[1] - prev_center[1])**2)
    
    # Convert pixels to meters
    distance_meters = distance_pixels / pixels_per_meter
    
    # Calculate speed (meters per second)
    speed = distance_meters * fps
    
    return speed

def process_video_segment(video_path, start_time, duration):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_time * fps)
    frames_to_process = min(int(duration * fps), total_frames - start_frame)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    vehicle_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    vehicle_speeds = {"car": [], "motorcycle": [], "bus": [], "truck": []}
    tracked_vehicles = []
    
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.3
    PIXELS_PER_METER = 20

    for _ in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_height = frame.shape[0]
        results = model(frame, verbose=False)
        
        current_detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                if class_name in vehicle_counts and conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0]
                    current_box = [int(x1), int(y1), int(x2), int(y2)]
                    current_center = get_box_center(current_box)
                    
                    new_vehicle = True
                    for tracked in tracked_vehicles:
                        if calculate_iou(current_box, tracked["box"]) > IOU_THRESHOLD:
                            new_vehicle = False
                            prev_center = tracked["center"]
                            
                            if is_moving_towards_camera(prev_center, current_center, frame_height):
                                tracked["moving_towards"] = True
                                
                                # Calculate and store speed
                                speed = calculate_speed(prev_center, current_center, fps, PIXELS_PER_METER)
                                tracked["speeds"].append(speed)
                            
                            tracked["box"] = current_box
                            tracked["center"] = current_center
                            break
                    
                    if new_vehicle:
                        tracked_vehicles.append({
                            "class": class_name,
                            "box": current_box,
                            "center": current_center,
                            "moving_towards": False,
                            "speeds": []
                        })
                    
                    current_detections.append({"class": class_name, "box": current_box})
    
    for tracked in tracked_vehicles:
        if tracked["moving_towards"]:
            vehicle_counts[tracked["class"]] += 1
            if tracked["speeds"]:
                avg_speed = sum(tracked["speeds"]) / len(tracked["speeds"])
                vehicle_speeds[tracked["class"]].append(avg_speed)
    
    cap.release()
    return vehicle_counts, vehicle_speeds


def traffic_load(vehicle_counts, vehicle_speeds):
    avg_speeds = {}
    for vehicle_type in vehicle_speeds:
        if vehicle_speeds[vehicle_type]:
            avg_speeds[vehicle_type] = sum(vehicle_speeds[vehicle_type]) / len(vehicle_speeds[vehicle_type])
        else:
            avg_speeds[vehicle_type] = 0
    
    weights = {"car": 1, "motorcycle": 0.5, "bus": 1.5, "truck": 2}
    total_weight = sum(weights[v] * vehicle_counts[v] for v in vehicle_counts)
    weighted_avg_speed = sum(weights[v] * avg_speeds[v] * vehicle_counts[v] for v in vehicle_counts) / total_weight if total_weight > 0 else 0
    C = vehicle_counts['car']
    T = vehicle_counts['bus'] + vehicle_counts['truck']
    M = vehicle_counts['motorcycle']
    load = ((1 * C) + (1.5 * T) + (0.5 * M)) - (weighted_avg_speed/5)
    return load


def optimize_traffic_light_timing(dir):
    vehicle_counts = process_video_segment(video_data[dir], start_time, duration)[0]
    C = vehicle_counts['car']
    T = vehicle_counts['bus'] + vehicle_counts['truck']
    M = vehicle_counts['motorcycle']
    timing = 20 + ((1 * C) + (1.5 * T) + (0.5 * M))
    if timing > 180:
        return 180
    return round(timing)



def get_load(args):
    direction, start_time, duration = args
    counts, speeds = process_video_segment(video_data[direction], start_time, duration)
    load = traffic_load(counts, speeds)
    return direction, load


# List of video paths and their corresponding start times
start_time = 0
duration = 2
video_data = {
        "north": "TrafficOptimization/input/north.mp4",
        "south": "TrafficOptimization/input/south.mp4",
        "east": "TrafficOptimization/input/east.mp4",
        "west": "TrafficOptimization/input/west.mp4"
    }


def main():
    arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)

    while True:
        with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(get_load, [(dir, 0, 2) for dir in video_data.keys()])
        
        loads = dict(results)
        loads = dict(sorted(loads.items(), key=lambda item: item[1],reverse=True))
        
        for dir, load in loads.items():
            timing = optimize_traffic_light_timing(dir)
            print(f"\n{dir}'s Green Light is ON for {timing} secs")
            
            arduino.write(f"{dir[0].upper()}:{timing}\n".encode())
            time.sleep(0.1)
            
            while True:
                if arduino.in_waiting > 0:
                    response = arduino.readline().decode().strip()
                    if response == "DONE":
                        break
                time.sleep(0.1)
        
        time.sleep(1)

if __name__ == "__main__":
    video_data = {
        "north": "TrafficOptimization/input/north.mp4",
        "south": "TrafficOptimization/input/south.mp4",
        "east": "TrafficOptimization/input/east.mp4",
        "west": "TrafficOptimization/input/west.mp4"
    }
    main()