import json
from dataclasses import asdict
from copy import deepcopy
from typing import List, Dict, Any
from dataclasses import dataclass, field
from all_modules import Attribute, Context, Detection, Prediction, TrackedObject


def json_to_context(json_str: str) -> Context:
    data = json.loads(json_str)

    tracked_objects = []
    for obj in data["model_result"]:
        # Tạo Detection
        detection = Detection(
            bbox=obj["bbox"],
            score=obj.get("score", 0.0),
            class_name=obj.get("class_name")
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
            frame_id=data["frame_id"],
            timestamp=data["timestamp"],
            detection=detection
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
        frame_id=data["frame_id"],
        timestamp=data["timestamp"],
        tracked_objects=tracked_objects
    )

    return context

if __name__ == "__main__":
    with open("simulated_model_input.json", "r") as f:
        example_json = f.read()
    context = json_to_context(example_json)
    context_dict = asdict(context)
    print(json.dumps(context_dict, indent=2))
    
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