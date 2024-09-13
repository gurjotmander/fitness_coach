import cv2
from ultralytics import YOLO
import pandas as pd

model = YOLO("C:/Users/gurjo/Documents/term 8/major project/fitness app/yolov8n-pose.pt")

video_path = "C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/videos/videoplayback.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}. Check the path and try again.")
    exit()

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    print("Error: Could not retrieve FPS from the video file.")
    exit()

seconds = round(frames/fps)

frame_total = 500 
i = 0
a = 0

all_data = []

while (cap.isOpened()):
    cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds/frame_total)*1000)))
    flag, frame = cap.read()

    if flag == False:
        break

    image_path = f'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/images/frames/img_{i}.jpg'
    cv2.imwrite(image_path, frame)

    # YOLOv8 Will detect video frame
    results = model(frame, verbose=False)

    for r in results:
        bound_box = r.boxes.xyxy  # get the bounding box on the frame
        conf = r.boxes.conf.tolist() # get the confident it is a human from a frame
        keypoints = r.keypoints.xyn.tolist() # get every human keypoint from a frame

        # Add a label for each detected person; modify this according to your task
        label = 'pose_label'  # Replace with actual label or class for the pose
        
        # Save every human that's detected from each image
        for index, box in enumerate(bound_box):
            if conf[index] > 0.75: # reduce blurry human image
                x1, y1, x2, y2 = box.tolist()
                pict = frame[int(y1):int(y2), int(x1):int(x2)]
                output_path = f'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/images/person/person_{a}.jpg'

                # Save the person image file name and label to CSV for later use
                data = {'image_name': f'person_{a}.jpg', 'label': label}

                # Initialize the x and y lists for each possible key
                for j in range(len(keypoints[index])):
                    data[f'x{j}'] = keypoints[index][j][0]
                    data[f'y{j}'] = keypoints[index][j][1]

                # Save human keypoint detected by YOLO model to CSV file
                all_data.append(data)
                cv2.imwrite(output_path, pict)
                a += 1

    i += 1

print(i-1, a-1)
cap.release()
cv2.destroyAllWindows()

# Combine all data dictionaries into a single DataFrame
df = pd.DataFrame(all_data)

# Save DataFrame to CSV file
csv_file_path = 'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/keypoints.csv'
df.to_csv(csv_file_path, index=False)