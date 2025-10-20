
from all_modules import *
import math
from copy import deepcopy
from typing import List, Tuple

# ZONE: Danger Zone (0, 0, 100, 100)
DANGER_ZONE = ZoneArea(zone_id="danger_zone", polygon=[0, 0, 100, 100])
EVENT_TRACKED_ID = Tracked_Id_Event()
# Hàm helper để di chuyển bbox
def move_bbox(bbox: List[float], dx: float, dy: float) -> List[float]:
    """Di chuyển bbox theo dx, dy"""
    return [
        bbox[0] + dx, 
        bbox[1] + dy, 
        bbox[2] + dx, 
        bbox[3] + dy
    ]

def get_person_object(bbox: List[float], frame_id: int, timestamp: int) -> TrackedObject:
    """Tạo đối tượng 'person' với vị trí cụ thể"""
    return TrackedObject(
        track_id=1,
        obj_name="person",
        predictions=[
            Prediction(
                frame_id=frame_id, 
                timestamp=timestamp,
                detection=Detection(bbox=bbox, score=0.9, class_name="person")
            )
        ]
    )

def create_context_with_person(frame_id: int, current_bbox: List[float], previous_ctx: Context = None) -> Context:
    """Tạo Context mới, duy trì lịch sử nếu có."""
    timestamp = frame_id * 1000
    
    # 1. TẠO prediction MỚI nhất (không tạo lại đối tượng TrackedObject hoàn chỉnh)
    new_prediction = Prediction(
        frame_id=frame_id, 
        timestamp=timestamp,
        detection=Detection(bbox=current_bbox, score=0.9, class_name="person")
    )
    
    # 2. Xử lý lịch sử
    current_predictions = [new_prediction]
    current_attribute = Attribute()
    
    if previous_ctx:
        old_person = next((obj for obj in previous_ctx.tracked_objects if obj.track_id == 1), None)
        if old_person:
             # Sao chép LỊCH SỬ CŨ và NỐI với prediction MỚI (CHỈ MỘT LẦN)
             current_predictions = old_person.predictions + current_predictions
             # Sao chép thuộc tính hiện tại (đặc biệt là timer)
             current_attribute = deepcopy(old_person.current_attribute)
             
    # 3. TẠO đối tượng person hoàn chỉnh với lịch sử đã được cập nhật
    person = TrackedObject(
        track_id=1,
        obj_name="person",
        predictions=current_predictions,
        current_attribute=current_attribute # Gán thuộc tính (timer, speed,...)
    )
    
    # 4. Tạo Context mới
    new_ctx = Context(
        frame_id=frame_id,
        timestamp=timestamp,
        tracked_objects=[person],
        global_vars={},
        events=[]
    )
             
    return new_ctx

# ZONE: Danger Zone (0, 0, 100, 100)
DANGER_ZONE = ZoneArea(zone_id="danger_zone", polygon=[0, 0, 100, 100])

def simulate_zone_entry_exit(num_frames: int = 5):
    """
    Mô phỏng người đi vào và đi ra khỏi vùng nguy hiểm.
    """
    print("\n" + "="*70)
    print(f"SIMULATION: ZONE ENTRY AND EXIT OVER {num_frames} FRAMES")
    print("="*70)
    
    # Kịch bản Bbox (Center of Zone: [50, 50, 50, 50])
    # Outside -> Inside -> Inside -> Inside -> Outside
    bbox_scenario = [
        [150, 150, 170, 170],  # Frame 1: OUTSIDE
        [40, 40, 60, 60],      # Frame 2: INSIDE (Entry Detected)
        [50, 50, 70, 70],      # Frame 3: INSIDE
        [60, 60, 80, 80],      # Frame 4: INSIDE (Timer should exceed here)
        [60, 60, 80, 80],      # Frame 4: INSIDE (Timer should exceed here)
        [120, 120, 140, 140]   # Frame 5: OUTSIDE (Exit Detected)
    ]

    # Pipeline 1: Phát hiện Entry và Bắt đầu Timer
    validate = Validate(tracked_id_event=EVENT_TRACKED_ID, name="0.Validate")
    filter_person = FilterObjects(labels=["person"], name="1.FilterPerson")
    check_inside = CheckZone(zones=[DANGER_ZONE], condition=ZoneCondition.IS_INSIDE, name="2.CheckInside")
    timer = Timer(timer_name="in_danger_zone", duration=2, name="3.TimerDangerZone") # Trigger sau 2 frames
    event_sender_timer = EventSender(event_name="ZONE_TIMEOUT", name="4.SendTimeoutAlert", tracked_id_event=EVENT_TRACKED_ID)

    validate.add_next(filter_person)
    filter_person.add_next(check_inside)
    check_inside.add_next(timer)
    timer.add_next(event_sender_timer)

    # Khởi tạo trạng thái ban đầu
    current_ctx = None
    all_events = []
    
    for frame_id in range(1, num_frames + 1):
        if frame_id > len(bbox_scenario):
            break
            
        current_bbox = bbox_scenario[frame_id - 1]
        
        # 1. Tạo Context mới với vị trí mới (duy trì lịch sử)
        new_ctx = create_context_with_person(frame_id, current_bbox, current_ctx)
        
        # 2. Thực thi pipeline
        print(f"\n--- Running Frame {frame_id} (Bbox: {current_bbox}) ---")
        
        results = validate.execute(new_ctx)
        
        # 3. Cập nhật trạng thái và Lưu lại sự kiện
        current_ctx = results[0] 
        all_events.extend(current_ctx.events)

        # In trạng thái để kiểm tra
        obj1 = next((obj for obj in current_ctx.tracked_objects if obj.track_id == 1), None)
        center = get_bbox_center(current_bbox)
        is_inside = (DANGER_ZONE.polygon[0] <= center[0] <= DANGER_ZONE.polygon[2] and 
                     DANGER_ZONE.polygon[1] <= center[1] <= DANGER_ZONE.polygon[3])
        
        timer_val = obj1.current_attribute.timers.get("in_danger_zone", 0) if obj1 else 0
        
        print(f"  -> Bbox Center: {center} | Inside Zone: {is_inside}")
        print(f"  -> Object 1 Timer: {timer_val}")

    
    print("\n" + "="*70)
    print(f"SIMULATION RESULT: ZONE ENTRY/EXIT")
    print("="*70)
    
    print(f"✓ Total events: {len(all_events)}")
    for event in all_events:
        print(f"  - {event.name} at Frame {event.frame_id}: {event.data}")


# Gọi hàm mô phỏng
simulate_zone_entry_exit(num_frames=6)