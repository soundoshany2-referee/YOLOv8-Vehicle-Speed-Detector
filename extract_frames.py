import os
import cv2

video_path = "video.mp4"
output_folder = "my_dataset_images"

os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(video_path)

count = 0
frame_id = 0

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  if count % 15 == 0:
    cv2.imwrite(f"{output_folder}/frame_{frame_id}.jpg", frame)
    frame_id += 1

  count += 1

cap.release()
print(f"Done{frame_id} images{output_folder}!")