# save as generate_simulated_contexts.py
import json
import random
from copy import deepcopy
from typing import List, Dict, Any

# Thông thường bạn đã có các dataclass: Detection, Pose, Segmentation, Attribute,
# Prediction, TrackedObject, Event, Context.
# Script này xuất JSON dưới dạng dict (serializable) tương thích cấu trúc Context bạn đưa.

def make_detection(x1, y1, w, h, score=0.95, class_name="person"):
    return {"bbox": [x1, y1, x1 + w, y1 + h], "score": score, "class_name": class_name}

def make_pose(x, y, conf=0.9):
    # simple 5 keypoints (head, shoulder, hip, left_hand, right_hand)
    return {"keypoints": [[x, y, conf], [x+5, y+20, conf], [x+8, y+45, conf],
                          [x-10, y+15, conf], [x+20, y+15, conf]], "score": conf}

def make_attribute(class_name="person", speed=None, direction=None, state=None, action=None,
                   color=None, ppe_status=None, nearby_objects=None, timers=None, group_id=None, altitude=None):
    return {
        "class_name": class_name,
        "speed": speed,
        "direction": direction,
        "state": state,
        "action": action,
        "color": color,
        "ppe_status": ppe_status or {},
        "nearby_objects": nearby_objects or {},
        "comovement": {},
        "timers": timers or {},
        "group_id": group_id,
        "altitude": altitude
    }

def generate_simulated_contexts(num_frames=15, out_path="simulated_contexts.json", seed=42):
    random.seed(seed)
    frames = []
    # initial positions for 4 tracks
    tracks_state = {
        1: {"x": 100, "y": 200, "vx": 5, "vy": 0, "class":"person"},   # moving worker (may miss PPE vest sometimes)
        2: {"x": 300, "y": 210, "vx": 0, "vy": 0, "class":"person"},   # stationary worker
        3: {"x": 0,   "y": 400, "vx": 40, "vy": 0, "class":"vehicle"},  # vehicle crossing line at frame ~6
        4: {"x": 150, "y": 250, "vx": 0, "vy": 0, "class":"person"}    # raises hand then falls (miss frame sim)
    }

    # helper: sometimes produce miss (no detection) to simulate track loss
    def should_miss(track_id, frame_id):
        # custom misses: track4 misses at frame 8, track1 misses at 11
        if track_id == 4 and frame_id == 8: 
            return True
        if track_id == 1 and frame_id == 11:
            return True
        # random small chance of miss
        return random.random() < 0.02

    for f in range(1, num_frames + 1):
        timestamp = 1696500000 + f * 0.04  # arbitrary increasing timestamp (float ok)
        tracked_objects = []

        # update states
        for tid, s in tracks_state.items():
            # move
            s["x"] += s["vx"]
            s["y"] += s["vy"]

        # Simulate events at certain frames
        events = []
        # frame 3: worker_1 missing vest -> ppe_violation
        if f == 3:
            events.append({"name":"ppe_violation", "frame_id": f, "timestamp": int(timestamp),
                           "data": {"track_id": 1, "missing_item": "vest"}})
        # frame 6: vehicle crosses line
        if f == 6:
            events.append({"name":"crossing_line", "frame_id": f, "timestamp": int(timestamp),
                           "data": {"track_id": 3, "line_id":"line_A"}})
        # frame 9: track4 fall event (after raise hand at 7)
        if f == 9:
            events.append({"name":"fall_detected", "frame_id": f, "timestamp": int(timestamp),
                           "data": {"track_id": 4}})
        # frame 11: lost tracking for track1
        if f == 11:
            events.append({"name":"track_lost", "frame_id": f, "timestamp": int(timestamp),
                           "data": {"track_id": 1}})
        # frame 13: track1 re-appears, ppe_ok
        if f == 13:
            events.append({"name":"ppe_ok", "frame_id": f, "timestamp": int(timestamp),
                           "data": {"track_id": 1}})

        # Build tracked objects entries (some may be missing predictions to simulate misses)
        # track 1
        for tid in [1,2,3,4]:
            base = tracks_state[tid]
            obj_name = f"obj_{tid}" if base["class"] != "person" else f"worker_{tid}"
            current_attr = deepcopy(make_attribute(
                class_name=base["class"],
                speed=round((base["vx"]**2 + base["vy"]**2)**0.5, 2),
                direction=0 if base["vx"] >= 0 else 180,
                state=("moving" if base["vx"] != 0 else "stationary"),
                action=("walking" if base["vx"] != 0 else "standing"),
                color=("blue" if tid==1 else "orange" if tid==2 else "gray" if tid==3 else "green"),
                ppe_status={"helmet": True, "vest": (False if (tid==1 and f in (3,4,5)) else True)} if base["class"]=="person" else None,
                nearby_objects={str(o): round(abs(tracks_state[o]["x"]-base["x"])+abs(tracks_state[o]["y"]-base["y"]),2)
                                for o in tracks_state if o!=tid and random.random()<0.6},
                timers={"loiter": round(max(0, f-5)/2.0,2)} if tid==2 else {}
            ))
            # build predictions: might be missing if should_miss
            predictions = []
            if not should_miss(tid, f):
                det = make_detection(base["x"], base["y"], 60 if base["class"]=="person" else 120, 180 if base["class"]=="person" else 60,
                                     score=round(random.uniform(0.85, 0.99), 2), class_name=base["class"])
                pose = make_pose(base["x"] + 10, base["y"] + 10, conf=round(random.uniform(0.7, 0.98),2)) if base["class"]=="person" else None
                seg = {"points":[base["x"], base["y"], base["x"]+30, base["y"]+40], "score": round(random.uniform(0.7,0.98),2)}
                attr = deepcopy(current_attr)
                # special behaviours for track4: raise hand at frame 7 then fall at 9
                if tid == 4 and f == 7:
                    attr["action"] = "raise_hand"
                    attr["timers"]["raise_hand_duration"] = 2.1
                if tid == 4 and f == 9:
                    attr["action"] = "falling"
                    attr["state"] = "falling"
                predictions.append({
                    "frame_id": f,
                    "timestamp": int(timestamp),
                    "detection": det,
                    "pose": pose,
                    "segmentation": seg,
                    "attribute": attr
                })
            else:
                # simulate missing detection -> add a prediction with None detection (or skip predictions entirely)
                # we'll add an explicit prediction with detection=null to indicate missed detection
                predictions.append({
                    "frame_id": f,
                    "timestamp": int(timestamp),
                    "detection": None,
                    "pose": None,
                    "segmentation": None,
                    "attribute": deepcopy(current_attr)
                })

            tracked_objects.append({
                "track_id": tid,
                "obj_name": obj_name,
                "current_attribute": current_attr,
                "predictions": predictions
            })

        frame_ctx = {
            "frame_id": f,
            "timestamp": int(timestamp),
            "tracked_objects": tracked_objects,
            "global_vars": {"zone_temperature": round(25 + random.random()*5,2), "alert_threshold": 0.8},
            "events": events
        }
        frames.append(frame_ctx)

    # write file
    with open(out_path, "w") as fw:
        json.dump(frames, fw, indent=2)
    print(f"Wrote {len(frames)} frames to {out_path}")
    return out_path

# --- Loaders ---
def load_simulated_contexts_json(file_path: str):
    """Load list of contexts (dicts) from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# If you want to parse each dict into your dataclasses, reuse your parse functions (see previous code).
def load_as_objects(file_path: str, parse_single_context_fn):
    """
    parse_single_context_fn: a function that takes a dict (one context) and returns Context dataclass
    """
    raw = load_simulated_contexts_json(file_path)
    return [parse_single_context_fn(ctx_dict) for ctx_dict in raw]

# --- Example usage ---
if __name__ == "__main__":
    path = generate_simulated_contexts(num_frames=15, out_path="simulated_contexts.json", seed=123)
    raw_frames = load_simulated_contexts_json(path)
    print(json.dumps(raw_frames[0], indent=2))
