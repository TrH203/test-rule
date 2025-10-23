import cv2
import json
import time
import numpy as np
import argparse
BOX_SIZE = 50  # bounding box width/height
clicked_pos = None
drawed = []

def draw_objects(frame, detections):
    """Draw bounding boxes and labels for detections."""
    for det in detections:
        x, y = det["bbox"][:2]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + BOX_SIZE, y + BOX_SIZE), color, 2)
        label = f'{det["class_name"]} #{det["tracked_id"]}'
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def get_click(event, x, y, flags, param):
    """Mouse click event to get coordinates."""
    global clicked_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pos = (x, y)
        drawed.append(clicked_pos)


def visualize_results(results):
    """Visualize saved detection results."""
    print("\n▶ Visualizing results (press ESC to stop)")
    for frame_info in results:
        frame = 255 * np.ones((400, 600, 3), dtype=np.uint8)
        draw_objects(frame, frame_info["model_result"])
        cv2.putText(frame, f"Frame {frame_info['frame_id']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        cv2.imshow("Visualization", frame)
        if cv2.waitKey(500) & 0xFF == 27:  # ESC to exit early
            break
    cv2.destroyAllWindows()


def main(file_name):
    global clicked_pos
    num_objects = int(input("Enter number of objects: "))
    class_names = [input(f"Class name for object {i + 1}: ") for i in range(num_objects)]
    num_frames = int(input("Enter number of frames: "))
    timestamp = 0
    results = []

    for frame_id in range(1, num_frames + 1):
        frame = 255 * np.ones((400, 600, 3), dtype=np.uint8)
        frame_dets = []
        timestamp += 1000
        # Drawed
        for pos in drawed:
            cv2.rectangle(frame, pos, (pos[0] + BOX_SIZE, pos[1] + BOX_SIZE), (222, 222, 222), 1)
            
        for obj_idx, cls in enumerate(class_names):
            cv2.imshow("Frame", frame)
            cv2.setMouseCallback("Frame", get_click)
            print(f"Frame {frame_id}, Object {cls} (click position)")

            clicked_pos = None
            while clicked_pos is None:
                if cv2.waitKey(20) & 0xFF == 27:
                    break

            if clicked_pos is None:
                continue

            x, y = clicked_pos
            bbox = [x, y, x + BOX_SIZE, y + BOX_SIZE]

            det = {
                "tracked_id": obj_idx + 1,
                "bbox": bbox,
                "score": round(0.8 + 0.2 * (obj_idx / num_objects), 2),
                "class_name": cls,
                "keypoints": [],
                "segmentation": [],
                "color": "green",
                "state": "moving",
                "action": None
            }
            frame_dets.append(det)

            draw_objects(frame, frame_dets)
            cv2.imshow("Frame", frame)
            cv2.waitKey(200)

        results.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "model_result": frame_dets
        })

    cv2.destroyAllWindows()

    # Save JSON file
    with open(file_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to {file_name}")

    # Prompt visualization
    print("Press 'v' to visualize result or any other key to quit...")
    prompt = 255 * np.ones((200, 400, 3), dtype=np.uint8)
    cv2.putText(prompt, "Press 'v' to visualize", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("Prompt", prompt)
    key = cv2.waitKey(0)
    cv2.destroyWindow("Prompt")

    if key == ord('v'):
        visualize_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file and print contexts.")
    parser.add_argument("-o", required=True, help="Path to the out JSON file")
    args = parser.parse_args()
    main(file_name=args.o)

