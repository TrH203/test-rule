import cv2
import json
import time
import numpy as np
from typing import List
import argparse
BOX_SIZE = 50  # bounding box width/height
from all_modules import ZoneArea

def draw_objects(frame, detections):
    """Draw bounding boxes and labels for detections."""
    for det in detections:
        x, y = det["bbox"][:2]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + BOX_SIZE, y + BOX_SIZE), color, 2)
        label = f'{det["class_name"]} #{det["tracked_id"]}'
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_zone(frame, zone: ZoneArea):
    """Draw zone area on the frame."""
    pts = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, zone.zone_id, tuple(pts[0][0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def visualize_results(results, aditional_boxes: List[ZoneArea] = None):
    """Visualize saved detection results."""
    print("\n Visualizing results (press ESC to stop)")
    for frame_info in results:
        frame = 255 * np.ones((400, 600, 3), dtype=np.uint8)
        if aditional_boxes:
            for i, _ in enumerate(aditional_boxes):
                draw_zone(frame, aditional_boxes[i])
        draw_objects(frame, frame_info["model_result"])
        cv2.putText(frame, f"Frame {frame_info['frame_id']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        cv2.imshow("Visualization", frame)
        if cv2.waitKey(500) & 0xFF == 27:  # ESC to exit early
            break
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize detection results from a JSON file.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file")
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    # visualize_results(results)
    visualize_results(results=results, aditional_boxes=[ZoneArea(zone_id="danger_zone", polygon=[0, 0, 200, 200])])
