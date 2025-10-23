
from all_modules import *
import math
from copy import deepcopy
from typing import List, Tuple
from convert_input_model_to_context import json_to_context

EVENT_TRACKED_ID = Tracked_Id_Event()

# ZONE: Danger Zone (0, 0, 100, 100)
DANGER_ZONE = ZoneArea(zone_id="danger_zone", polygon=[0, 0, 200, 200])

def simulate_zone_entry_exit(data_file: str = None):
    """
    Mô phỏng người đi vào và đi ra khỏi vùng nguy hiểm.
    """
    
    # Pipeline 1: Phát hiện Entry và Bắt đầu Timer
    validate = Validate(tracked_id_event=EVENT_TRACKED_ID, name="0.Validate")
    filter_vehicle = FilterObjects(labels=["vehicle"], name="1.FilterPerson")
    check_speed = AnalyzingSpeed(condition=SpeedCondition.IS_FASTER_THAN, value=1, time_window=2, name="2.CheckSpeed")
    timer = Timer(timer_name="in_danger_zone", duration=2, name="3.TimerDangerZone") # Trigger sau 2 frames
    event_sender_timer = EventSender(event_name="ZONE_TIMEOUT", name="4.SendTimeoutAlert", tracked_id_event=EVENT_TRACKED_ID)

    validate.add_next(filter_vehicle)
    filter_vehicle.add_next(check_speed)
    check_speed.add_next(timer)
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

    
    
    print("\n" + "="*70)
    print(f"SIMULATION RESULT: ZONE ENTRY/EXIT")
    print("="*70)
    
    print(f"✓ Total events: {len(all_events)}")
    for event in all_events:
        print(f"  - {event.name} at Frame {event.frame_id}: {event.data}")


# Gọi hàm mô phỏng
simulate_zone_entry_exit(data_file="simulation-in-out-dangerzone2.json") 