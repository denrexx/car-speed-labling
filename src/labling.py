import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


# 1st step: Get FPS, w and h
folder = Path(__file__).parent.parent
video_path = folder / "ScreenRecorderProject49.mp4"
model = YOLO(folder / "yolo26s.pt")
video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    raise RuntimeError("Unable to open the video")

fps = video.get(cv2.CAP_PROP_FPS)
frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
shows_data = True

if shows_data:
    print(f"FPS: {fps}")
    print(f"Frame: {frame}")
    print(f"Width: {w}")
    print(f"Height: {h}")


# Click six points: 2 for count line, 2 for speed A and 2 for speed B.
def choose_lines(image):
    points = []
    names = ["Count line", "Speed line A", "Speed line B"]

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
            points.append((x, y))

    cv2.namedWindow("Choose lines")
    cv2.setMouseCallback("Choose lines", mouse)

    while True:
        preview = image.copy()
        for i, point in enumerate(points):
            cv2.circle(preview, point, 6, (0, 255, 255), -1)
            if i % 2 == 1:
                cv2.line(preview, points[i - 1], point, (0, 255, 255), 2)
                cv2.putText(preview, names[i // 2], points[i - 1], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(preview, "Click 6 points. Enter - save, Q - exit", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Choose lines", preview)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            return None
        if key == 13 and len(points) == 6:
            return points


# Set True once, choose lines by hand, then paste printed coordinates below.
set_lines = False

# These values must be replaced after manual selection.
count_line = [(1350, 946), (1350, 80)]
line_a = [(180, 939), (230, 80)]
line_b = [(1231, 941), (1212, 80)]

# Real distance between line A and B. Set it by hand in metres.
distance_m = 4.8

if set_lines:
    video.set(cv2.CAP_PROP_POS_FRAMES, int(13.1 * fps))
    good, image = video.read()
    if not good:
        raise RuntimeError("Unable to read video frame")

    points = choose_lines(image)
    video.release()
    cv2.destroyAllWindows()

    if points is not None:
        print("count_line =", points[0:2])
        print("line_a =", points[2:4])
        print("line_b =", points[4:6])
    raise SystemExit("Paste the coordinates, set set_lines = False and set distance_m")


def line_side(point, line):
    return (line[1][0]-line[0][0])*(point[1]-line[0][1])-(line[1][1]-line[0][1])*(point[0]-line[0][0])


def has_crossed(old_side, new_side):
    return old_side * new_side < 0


def get_tracked_boxes(boxes):
    if boxes is None:
        return []

    ids = boxes.id
    if ids is None:
        return []

    track_ids = [int(track_id) for track_id in ids.tolist()]
    return [(boxes[index], track_id) for index, track_id in enumerate(track_ids)]


def new_record(track_id):
    return {
        "id": track_id,
        "time": "",
        "start": "",
        "end": "",
        "delta": "",
        "distance": distance_m,
        "speed": ""
    }


# 3rd step: tracking and trajectories
tracks = defaultdict(lambda: deque(maxlen=30))
last_sides = defaultdict(dict)
counted_ids = set()
start_times = {}
records = {}
last_speeds = {}

video.set(cv2.CAP_PROP_POS_FRAMES, 0)
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
writer = cv2.VideoWriter(str(folder / "result.mp4"), fourcc, fps, (w, h))
if not writer.isOpened():
    raise RuntimeError("Unable to create result video")

frame_index = 0

while True:
    good, img = video.read()
    if not good:
        break

    result = model.track(
        img,
        persist=True,
        tracker=str(folder / "bytetrack_custom.yaml"),
        imgsz=1280,
        classes=[2],
        conf=0.1,
        iou=0.8,
        verbose=False
    )[0]
    time_now = frame_index / fps

    tracked_boxes = get_tracked_boxes(result.boxes)
    if tracked_boxes:

        for box, track_id in tracked_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            center = ((x1+x2)//2, y2)
            tracks[track_id].append(center)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 166, 255), 2)
            cv2.putText(img, f"car {confidence:.2f} ID: {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 166, 255), 2)

            points = list(tracks[track_id])
            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], (0, 255, 0), 2)

            current_sides = {
                "count": line_side(center, count_line),
                "a": line_side(center, line_a),
                "b": line_side(center, line_b)
            }

            for name, current_side in current_sides.items():
                old_side = last_sides[track_id].get(name)
                if old_side is not None and has_crossed(old_side, current_side):
                    direction = "forward" if old_side > current_side else "backward"

                    if name == "count" and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        if track_id not in records:
                            records[track_id] = new_record(track_id)
                        records[track_id]["time"] = round(time_now, 2)

                    if name == "a":
                        start_times[track_id] = (time_now, direction)

                    if name == "b" and track_id in start_times:
                        time_a, direction_a = start_times[track_id]
                        delta_t = time_now - time_a
                        if delta_t > 0 and direction == direction_a:
                            speed = 3.6 * distance_m / delta_t
                            last_speeds[track_id] = speed
                            if track_id not in records:
                                records[track_id] = new_record(track_id)
                            records[track_id]["start"] = round(time_a, 2)
                            records[track_id]["end"] = round(time_now, 2)
                            records[track_id]["delta"] = round(delta_t, 2)
                            records[track_id]["speed"] = round(speed, 1)
                            del start_times[track_id]

                last_sides[track_id][name] = current_side

            if track_id in last_speeds:
                cv2.putText(img, f"{last_speeds[track_id]:.1f} km/h", (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.line(img, count_line[0], count_line[1], (0, 0, 255), 3)
    cv2.line(img, line_a[0], line_a[1], (255, 255, 0), 2)
    cv2.line(img, line_b[0], line_b[1], (255, 255, 0), 2)
    cv2.putText(img, f"Cars: {len(counted_ids)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Direction: forward / backward", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    writer.write(img)
    frame_index += 1

video.release()
writer.release()

saved_records = [records[track_id] for track_id in counted_ids]
saved_records.sort(key=lambda item: item["time"])
for car_id, record in enumerate(saved_records, start=1):
    record["id"] = car_id

with open(folder / "data" / "cars.csv", "w", newline="", encoding="utf-8") as file:
    columns = ["id", "time", "start", "end", "delta", "distance", "speed"]
    csv_writer = csv.DictWriter(file, fieldnames=columns)
    csv_writer.writeheader()
    csv_writer.writerows(saved_records)

print("Cars:", len(counted_ids))
saved_speeds = [record["speed"] for record in saved_records if record["speed"] != ""]
if saved_speeds:
    print("Average speed:", round(sum(saved_speeds) / len(saved_speeds), 1), "km/h")
print("Saved: result.mp4 and data/cars.csv")
