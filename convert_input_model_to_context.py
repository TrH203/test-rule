import json
from dataclasses import asdict
from copy import deepcopy
from typing import List, Dict, Any
from dataclasses import dataclass, field
from all_modules import *
import argparse
import json
from dataclasses import asdict

def json_to_context(json_str: str) -> List[Context]:
  with open(json_str, "r") as f:
    example_json = f.read()
    
  data = json.loads(example_json)
  list_ctx = []
  for raw_ctx in data:
    tracked_objects = []
    for obj in raw_ctx["model_result"]:
        # Tạo Detection
        detection = Detection(
            bbox=obj["bbox"],
            score=obj.get("score", 0.0),
            class_name=obj.get("class_name")
        )
        pose = Pose(
          keypoints=obj.get("keypoints", []),
          score=obj.get("pose_score", 0.0)
        )
        segmentation = Segmentation(
          points=obj.get("segmentation", []),
          score=obj.get("segmentation_score", 0.0)
        )
        # Tạo Attribute
        attr = Attribute(
            class_name=obj.get("class_name"),
            color=obj.get("color"),
            state=obj.get("state"),
            action=obj.get("action")
        )

        # Tạo Prediction
        prediction = Prediction(
            frame_id=raw_ctx["frame_id"],
            timestamp=raw_ctx["timestamp"],
            detection=detection,
            segmentation=segmentation,
            pose=pose,
            attribute=None
        )

        # Tạo TrackedObject
        tracked_obj = TrackedObject(
            track_id=obj["tracked_id"],
            obj_name=obj.get("class_name"),
            current_attribute=attr,
            predictions=[prediction]
        )

        tracked_objects.append(tracked_obj)

    # Tạo Context
    context = Context(
        frame_id=raw_ctx["frame_id"],
        timestamp=raw_ctx["timestamp"],
        tracked_objects=tracked_objects
    )
    list_ctx.append(context)

  return list_ctx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file and print contexts.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file")
    args = parser.parse_args()

    contexts = json_to_context(args.input)

    for context in contexts:
        context_dict = asdict(context) if not isinstance(context, dict) else context
        print(json.dumps(context_dict, indent=2))
        print("--------------------------")
    
"""
{
  "frame_id": 1,
  "timestamp": 1000,
  "tracked_objects": [
    {
      "track_id": 1,
      "obj_name": "staff",
      "current_attribute": {
        "class_name": "staff",
        "speed": null,
        "direction": null,
        "state": null,
        "action": null,
        "color": null,
        "ppe_status": null,
        "nearby_objects": null,
        "comovement": null,
        "timers": {},
        "group_id": null,
        "altitude": null
      },
      "predictions": [
        {
          "frame_id": 1,
          "timestamp": 1000,
          "detection": {
            "bbox": [
              10,
              20,
              40,
              20
            ],
            "score": 0.8,
            "class_name": "staff"
          },
          "pose": null,
          "segmentation": null,
          "attribute": null
        }
      ]
    },
    {
      "track_id": 2,
      "obj_name": "customer",
      "current_attribute": {
        "class_name": "customer",
        "speed": null,
        "direction": null,
        "state": null,
        "action": null,
        "color": null,
        "ppe_status": null,
        "nearby_objects": null,
        "comovement": null,
        "timers": {},
        "group_id": null,
        "altitude": null
      },
      "predictions": [
        {
          "frame_id": 1,
          "timestamp": 1000,
          "detection": {
            "bbox": [
              10,
              20,
              40,
              20
            ],
            "score": 0.8,
            "class_name": "customer"
          },
          "pose": null,
          "segmentation": null,
          "attribute": null
        }
      ]
    }
  ],
  "global_vars": {},
  "events": []
}"""