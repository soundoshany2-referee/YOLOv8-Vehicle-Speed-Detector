import cv2
from ultralytics import YOLO


class SpeedLine:

  def __init__(self, x1, x2, y):
    self.x1 = x1
    self.x2 = x2
    self.y = y

  def is_crossed(self, cx, prev_y, cur_y):
    if self.x1 <= cx <= self.x2:
      if (prev_y <= self.y <= cur_y) or (cur_y <= self.y <= prev_y):
        return True
    return False


class Vehicle:

  def __init__(self, track_id):
    self.track_id = track_id
    self.prev_y = None
    self.entry_frame = None
    self.speed_kmh = None

  def update(self, cx, cy, frame_num, line_a, line_b, fps, distance_m=10.0):
    if self.prev_y is not None:
      if self.entry_frame is None:
        if line_a.is_crossed(cx, self.prev_y, cy) or line_b.is_crossed(
            cx, self.prev_y, cy
        ):
          self.entry_frame = frame_num
      else:
        if (
            self.speed_kmh is None
            and (line_a.is_crossed(cx, self.prev_y, cy))
            or line_b.is_crossed(cx, self.prev_y, cy)
        ):
          exit_frame = frame_num
          frames_elapsed = abs(exit_frame - self.entry_frame)

          if frames_elapsed > 0:
            time_seconds = frames_elapsed / fps
            speed_m_s = distance_m / time_seconds
            self.speed_kmh = speed_m_s * 3.6

    self.prev_y = cy


class SpeedDetector:

  def __init__(
      self, model_path, video_path, output_path="output.mp4"
  ):
    self.model = YOLO(model_path)
    self.video_path = video_path
    self.output_path = output_path
    self.vehicles = {}

    cap = cv2.VideoCapture(self.video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()

    self.line_a = SpeedLine(x1=0, x2=w, y=int(h * 0.45))
    self.line_b = SpeedLine(x1=0, x2=w, y=int(h * 0.65))
    self.distance_m = 10.0

  def run(self):
    cap = cv2.VideoCapture(self.video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # إعداد أداة حفظ الفيديو
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

    frame_num = 0

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      frame_num += 1

      results = self.model.track(
          frame, persist=True, tracker="bytetrack.yaml", conf=0.3
      )
      boxes = results[0].boxes

      if boxes is not None and boxes.id is not None:
        xyxy_list = boxes.xyxy.int().tolist()
        track_ids = boxes.id.int().tolist()

        for (x1, y1, x2, y2), track_id in zip(xyxy_list, track_ids):
          cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

          if track_id not in self.vehicles:
            self.vehicles[track_id] = Vehicle(track_id)

          vehicle = self.vehicles[track_id]
          vehicle.update(
              cx,
              cy,
              frame_num,
              self.line_a,
              self.line_b,
              fps,
              self.distance_m,
          )

          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

          label = f"ID: {track_id}"
          if vehicle.speed_kmh is not None:
            label += f" | {vehicle.speed_kmh:.1f} km/h"

          cv2.putText(
              frame,
              label,
              (x1, y1 - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.6,
              (0, 255, 0),
              2,
          )

      cv2.line(
          frame,
          (self.line_a.x1, self.line_a.y),
          (self.line_a.x2, self.line_a.y),
          (255, 0, 0),
          2,
      )
      cv2.line(
          frame,
          (self.line_b.x1, self.line_b.y),
          (self.line_b.x2, self.line_b.y),
          (0, 0, 255),
          2,
      )

      # كتابة الفريم المعدل في ملف الفيديو الناتج
      out.write(frame)

      cv2.imshow("Vehicle Speed Detector", frame)

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  detector = SpeedDetector(
      model_path="best.pt", video_path="video.mp4", output_path="output.mp4"
  )
  detector.run()