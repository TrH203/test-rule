from all_modules import *
import math
from copy import deepcopy
from typing import List, Tuple

def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Lấy tâm của bounding box"""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def move_bbox(bbox: List[float], dx: float, dy: float) -> List[float]:
    """Di chuyển bbox theo dx, dy"""
    return [
        bbox[0] + dx, 
        bbox[1] + dy, 
        bbox[2] + dx, 
        bbox[3] + dy
    ]

def get_base_context_for_movement():
    """Tạo context cơ bản cho mô phỏng di chuyển"""
    return Context(
        frame_id=1,
        timestamp=1000,
        tracked_objects=[
            TrackedObject(
                track_id=1,
                obj_name="person",
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[100, 100, 120, 120], score=0.9, class_name="person"))
                ]
            ),
            TrackedObject(
                track_id=2,
                obj_name="vehicle",
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[110, 110, 150, 150], score=0.85, class_name="vehicle"))
                ]
            )
        ]
    )
    
def simulate_co_movement(num_frames: int = 5):
    """
    Mô phỏng chuỗi frame liên tục để kiểm tra node CheckCoMovement.
    """
    print("\n" + "="*70)
    print(f"SIMULATION: CO-MOVEMENT OVER {num_frames} FRAMES")
    print("="*70)
    
    # Định nghĩa pipeline
    filter_objects = FilterObjects(labels=["person", "vehicle"], name="1.FilterPV")
    check_comovement = CheckCoMovement(time_window=3.0, 
                                       velocity_tolerance=5.0, # Cho phép sai khác vận tốc lên đến 5.0
                                       direction_tolerance=30.0, # Cho phép sai khác hướng lên đến 30 độ
                                       labels=["person", "vehicle"],
                                       name="2.DetectCoMovement")
    event_sender = EventSender(event_name="COMOVEMENT_DETECTED", name="3.SendAlert")
    
    # Kết nối
    # filter_objects.add_next(check_comovement)
    # check_comovement.add_next(event_sender)
    
    check_comovement.add_next(filter_objects)
    filter_objects.add_next(event_sender)
    
    # Khởi tạo trạng thái ban đầu
    current_ctx = get_base_context_for_movement()
    all_events = []
    
    # Chuyển động giả lập (đơn vị pixel/frame)
    # Tương đương với: person (15, 5), vehicle (12, 5) -> Chênh lệch nhỏ
    movements = {
        1: (15.0, 5.0),  # person: Tốc độ cao
        2: (12.0, 5.0)   # vehicle: Tốc độ gần bằng, cùng hướng
    }
    
    for frame_id in range(1, num_frames + 1):
        # 1. Tạo Context mới và CẬP NHẬT VỊ TRÍ
        new_ctx = current_ctx.clone() 
        new_ctx.frame_id = frame_id
        new_ctx.timestamp = frame_id * 1000 # Giả sử 1 frame = 1 giây
        new_ctx.events = [] 
        
        updated_objects = []
        for obj in new_ctx.tracked_objects:
            if obj.track_id in movements:
                dx, dy = movements[obj.track_id]
                
                # Cập nhật bbox (chỉ lấy prediction cuối cùng của frame trước)
                last_pred = obj.predictions[-1]
                new_bbox = move_bbox(last_pred.detection.bbox, dx * (frame_id - last_pred.frame_id), dy * (frame_id - last_pred.frame_id))
                
                # Thêm prediction mới
                new_pred = Prediction(
                    frame_id=frame_id, 
                    timestamp=frame_id * 1000,
                    detection=Detection(bbox=new_bbox, score=0.9, class_name=obj.obj_name)
                )
                obj.predictions.append(new_pred)

            updated_objects.append(obj)
            
        new_ctx.tracked_objects = updated_objects
        
        # 2. Thực thi pipeline
        print(f"\n--- Running Frame {frame_id} (Timestamp: {new_ctx.timestamp}ms) ---")
        
        # Cần ít nhất 2 prediction để tính vận tốc (tức là từ frame 2 trở đi)
        if frame_id >= 2:
            results = filter_objects.execute(new_ctx)
            current_ctx = results[0] 
            all_events.extend(current_ctx.events)
            
            # In vận tốc để kiểm tra
            person = next((obj for obj in current_ctx.tracked_objects if obj.track_id == 1), None)
            vehicle = next((obj for obj in current_ctx.tracked_objects if obj.track_id == 2), None)
            
            if person and vehicle:
                # TÍNH TOÁN
                speed_p, dir_p = calculate_velocity(person)
                speed_v, dir_v = calculate_velocity(vehicle)
                # GÁN VÀO THUỘC TÍNH (Bước thiếu)
                if person.current_attribute is None:
                    person.current_attribute = Attribute()
                person.current_attribute.speed = speed_p
                person.current_attribute.direction = dir_p

                if vehicle.current_attribute is None:
                    vehicle.current_attribute = Attribute()
                vehicle.current_attribute.speed = speed_v
                vehicle.current_attribute.direction = dir_v
                
                # IN RA KẾT QUẢ
                print(f"  -> Person Speed: {speed_p:.2f} | Vehicle Speed: {speed_v:.2f}")
                print(f"  -> Person Dir: {dir_p:.2f} | Vehicle Dir: {dir_v:.2f}")
        else:
            print("  -> (Skipping CheckCoMovement - Not enough history)")
            current_ctx = new_ctx # Nếu không chạy, vẫn clone trạng thái mới nhất
    
    print("\n" + "="*70)
    print(f"SIMULATION RESULT: {num_frames} FRAMES")
    print("="*70)
    
    print(f"✓ Total events: {len(all_events)}")
    for event in all_events:
        print(f"  - {event.name} at Frame {event.frame_id}: {event.data}")

# Gọi hàm mô phỏng
simulate_co_movement(num_frames=10)