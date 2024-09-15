import cv2
from ultralytics import YOLO
import pandas as pd
import os

model = YOLO("C:/Users/gurjo/Documents/term 8/major project/fitness app/yolov8n-pose.pt")

videos = [
    "C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/videos/vid1.mp4",
    "C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/videos/vid2.mov",
    "C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/videos/vid3.mov"
]

all_data = []

for video in videos:
  cap = cv2.VideoCapture(video)
  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  seconds = frames / fps
  frame_total = 500  # Total number of frames to process
  step_size = max(1, int(frames / frame_total))

  i = 0
  a = 0

  try:

    while (cap.isOpened() and i < frame_total):

      # Move to the next frame to read
      cap.set(cv2.CAP_PROP_POS_FRAMES, i * step_size)
      flag, frame = cap.read()

      if not flag:
        break

      # Save video frame as image
      video_name = os.path.basename(video).split('.')[0]  # Extract filename of video
      image_path = f'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/images/frames/img_{video_name}_{i}.jpg'
      cv2.imwrite(image_path, frame)

      # YOLOv8 Will detect video frame
      results = model(frame, verbose=False)

      for r in results:
        bound_box = r.boxes.xyxy  # get the bounding box on the frame
        conf = r.boxes.conf.tolist()  # get the confidence it is a human from a frame
        keypoints = r.keypoints.xyn.tolist()  # get every human keypoint from a frame

        label = 'pose'

        # Save every human that's detected from each image
        for index, box in enumerate(bound_box):
          if conf[index] > 0.75:  # reduce blurry human image
            x1, y1, x2, y2 = map(int, box.tolist())
            pict = frame[y1:y2, x1:x2]
            output_path = f'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/images/person/person_{video_name}_{a}.jpg'

            # Save the person image file name and label to CSV for later use
            data = {'image_name': f'person_{video_name}_{a}.jpg', 'label': label}

            # Initialize the x and y lists for each possible key
            for j in range(len(keypoints[index])):
              data[f'x{j}'] = keypoints[index][j][0] / frame.shape[1]
              data[f'y{j}'] = keypoints[index][j][1] / frame.shape[0]

            # Save human keypoint detected by YOLO model to CSV file
            all_data.append(data)
            cv2.imwrite(output_path, pict)
            a += 1

      i += 1

  finally:
    cap.release()
    cv2.destroyAllWindows()

# Combine all data dictionaries into a single DataFrame
df = pd.DataFrame(all_data)

# Save DataFrame to CSV file
csv_file_path = 'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/keypoints.csv'
df.to_csv(csv_file_path, index=False)
