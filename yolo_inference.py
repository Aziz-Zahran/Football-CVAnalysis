from ultralytics import YOLO

model = YOLO('models/best.pt');

results = model.predict('input_videos/test(33).mp4',save=True)
print(results[0])

print("=================================")

for box in results[0].boxes:
    print(box)
# for result in results:
#     boxes = result.boxes
#     for box in boxes:
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         class_id = box.cls[0].item()
#         confidence = box.conf[0].item()
#         print(f"Box: {x1}, {y1}, {x2}, {y2}, Class: {class_id}, Confidence: {confidence}")