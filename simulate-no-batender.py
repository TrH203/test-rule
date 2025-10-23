
from convert_input_model_to_context import json_to_context
from all_modules import *
import math
from copy import deepcopy
from typing import List, Tuple
import argparse

# ZONE: Danger Zone (0, 0, 100, 100)
DANGER_ZONE = ZoneArea(zone_id="danger_zone", polygon=[0, 0, 100, 100])
EVENT_TRACKED_ID = Tracked_Id_Event()

def simulate(num_frames: int = 5, data_file: str = None):
    
    # Pipeline 1: Phát hiện Entry và Bắt đầu Timer
    validate = Validate(tracked_id_event=EVENT_TRACKED_ID, name="0.Validate")
    
    filter_person1 = FilterObjects(labels=["customer"], name="1.1.FilterPerson")
    filter_person2 = FilterObjects(labels=["staff"], name="1.2.FilterPerson")
    
    check_pose = CheckPose(pose_condition=PoseCondition.RAISE_HAND, name="2.CheckPoseStanding")
    
    merge_node = MergeNode(name="3.MergeChecks")
    
    check_group = CheckGroup(min_group_size=2, max_group_size=5, max_distance=100, name="4.CheckGroupSize", labels=["customer", "staff"])
    
    timer = Timer(timer_name="in_danger_zone", duration=2, name="5.Timer") # Trigger sau 2 frames
    event_sender_timer = EventSender(event_name="6.SendEvent", tracked_id_event=EVENT_TRACKED_ID)
    # event_sender_timer = EventSender(event_name="6.SendEvent")
    

    validate.add_next(filter_person1)
    validate.add_next(filter_person2)
    
    filter_person1.add_next(check_pose)
    
    check_pose.add_next(merge_node)
    filter_person2.add_next(merge_node)
    
    
    merge_node.add_next(check_group)
    
    check_group.add_next(timer)    
    timer.add_next(event_sender_timer)

    # Khởi tạo trạng thái ban đầu
    contexts = json_to_context(data_file)
    current_ctx = None
    all_events = []
    
    
    for context in contexts:
            
        
        new_ctx = context
        
        # 2. Thực thi pipeline
        print(f"\n--- Running Frame {new_ctx.frame_id}")
        
        results = validate.execute(new_ctx)
        
        # 3. Cập nhật trạng thái và Lưu lại sự kiện
        current_ctx = results[0]
        all_events.extend(current_ctx.events)

    print("\n","="*10, end="")
    print("All Events Generated", end="")
    print("="*10)
    for event in all_events:
        print(f"\n{event.name} at Frame {event.frame_id}: {event.data}")
    print("\n", "="*50)


# Gọi hàm mô phỏng
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process a JSON file and print contexts.")
    # parser.add_argument("--input", required=True, help="Path to the input JSON file")
    # args = parser.parse_args()
    # simulate(data_file = args.input)
    simulate(data_file = "simulation-no-batender.json")
    