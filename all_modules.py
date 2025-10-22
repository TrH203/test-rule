from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from enum import Enum
from copy import deepcopy
import math


# ============================================================
# BASIC DATA TYPES
# ============================================================

@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    score: float = 0.0
    class_name: Optional[str] = None


@dataclass
class Pose:
    keypoints: List[List[float]]  # (n, 3): [x, y, conf]
    score: float = 0.0


@dataclass
class Segmentation:
    points: List[float]
    score: float = 0.0


@dataclass
class Attribute:
    class_name: Optional[str] = None
    speed: Optional[float] = None
    direction: Optional[float] = None  # degrees
    state: Optional[str] = None
    action: Optional[str] = None
    color: Optional[str] = None
    ppe_status: Optional[Dict[str, bool]] = None  # {"helmet": True, "vest": False}
    nearby_objects: Optional[Dict[int, float]] = None  # {track_id: distance}
    comovement: Optional[Dict[int, Tuple[float, float]]] = None  # {track_id: (velocity_diff, direction_diff)}
    timers: Dict[str, float] = field(default_factory=dict)  # {timer_name: duration}
    group_id: Optional[int] = None
    altitude: Optional[float] = None


@dataclass
class Prediction:
    frame_id: int
    timestamp: int
    detection: Optional[Detection] = None
    pose: Optional[Pose] = None
    segmentation: Optional[Segmentation] = None
    attribute: Optional[Attribute] = None


@dataclass
class TrackedObject:
    track_id: int
    obj_name: Optional[str] = None
    current_attribute: Optional[Attribute] = field(default_factory=Attribute)
    predictions: List[Prediction] = field(default_factory=list)


@dataclass
class Event:
    name: str
    frame_id: int
    timestamp: int
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZoneArea:
    zone_id: str
    polygon: List[float]  # [x1, y1, x2, y2] for simplicity


# ============================================================
# ENUMS
# ============================================================

class PPEType(Enum):
    HELMET = "helmet"
    VEST = "vest"
    SAFETY_SHOES = "safety_shoes"


class PPECondition(Enum):
    IS_WEARING = "is_wearing"
    IS_NOT_WEARING = "is_not_wearing"


class ItemType(Enum):
    PHONE = "phone"
    ALCOHOL_TEST = "alcohol_test"


class PoseCondition(Enum):
    WALKING = 'walking'
    STANDING = 'standing'
    SITTING = 'sitting'
    LYING = 'lying'
    RAISE_HAND = 'raise_hand'
    LIFTING = 'lifting'
    FALLING = 'falling'


class MovingCondition(Enum):
    MOVING = 'moving'
    STATIONARY = 'stationary'


class SpeedCondition(Enum):
    IS_FASTER_THAN = "is_faster_than"
    IS_SLOWER_THAN = "is_slower_than"


class ZoneCondition(Enum):
    IS_INSIDE = "is_inside"
    IS_OUTSIDE = "is_outside"
    CROSSES_LINE = "crosses_line"


class AltitudeCondition(Enum):
    IS_HIGHER = 'is_higher'
    IS_LOWER = 'is_lower'


class ImpactSensitivity(Enum):
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'


# ============================================================
# CONTEXT - Luồng dữ liệu thống nhất
# ============================================================

@dataclass
class Context:
    """Context chứa toàn bộ dữ liệu đi qua pipeline"""
    frame_id: int
    timestamp: int
    tracked_objects: List[TrackedObject] = field(default_factory=list)
    global_vars: Dict[str, Any] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    
    def clone(self):
        """Tạo bản sao để xử lý song song"""
        return Context(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            tracked_objects=deepcopy(self.tracked_objects),
            global_vars=deepcopy(self.global_vars),
            events=deepcopy(self.events)
        )

class Tracked_Id_Event:
    """Lưu trữ mapping giữa tracked_id và event đã xử lý"""
    def __init__(self):
        self.mapper: Dict[str, str] = {}
        
    def update(self, track_id: str, event_name: str):
        self.mapper[track_id] = event_name
    
    def check(self, tracked_id, event_name) -> bool:
        return self.mapper.get(tracked_id) == event_name
    
    def check(self, track_id: str) -> bool:
        return track_id in self.mapper
# ============================================================
# BASE NODE CLASS
# ============================================================

class Node:
    """Base class cho tất cả các node trong pipeline"""
    
    def __init__(self, name: str = "Node"):
        self.name = name
        self.next_nodes: List['Node'] = []
    
    def add_next(self, node: 'Node'):
        """Thêm node tiếp theo"""
        if isinstance(node, list):
            self.next_nodes.extend(node)
        else:
            self.next_nodes.append(node)
        return node
    
    def process(self, ctx: Context) -> Context:
        """Xử lý context và trả về context mới"""
        return ctx
    
    def execute(self, ctx: Context) -> List[Context]:
        """Thực thi node và truyền kết quả cho next nodes"""
        processed_ctx = self.process(ctx)
        
        if not self.next_nodes:
            return [processed_ctx]
        
        results = []
        for next_node in self.next_nodes:
            results.extend(next_node.execute(processed_ctx))
        
        return results


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Lấy tâm của bounding box"""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def calculate_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """Tính khoảng cách Euclidean giữa 2 bbox centers"""
    c1 = get_bbox_center(bbox1)
    c2 = get_bbox_center(bbox2)
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def calculate_velocity(obj: TrackedObject, time_window: float = 1.0) -> Tuple[float, float]:
    """
    Tính vận tốc và hướng di chuyển
    Returns: (speed, direction_degrees)
    """
    time_window *= 1000  # convert to milliseconds
    if len(obj.predictions) <= 2:
        return 0.0, 0.0
    
    recent_preds = [p for p in obj.predictions if p.timestamp >= obj.predictions[-1].timestamp - time_window]
    if len(recent_preds) < 2:
        return 0.0, 0.0
    
    bbox1 = recent_preds[0].detection.bbox
    bbox2 = recent_preds[-1].detection.bbox
    
    c1 = get_bbox_center(bbox1)
    c2 = get_bbox_center(bbox2)
    
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    
    distance = math.sqrt(dx**2 + dy**2)
    time_diff = (recent_preds[-1].timestamp - recent_preds[0].timestamp) / 1000.0  # convert to seconds
    
    speed = distance / time_diff if time_diff > 0 else 0.0
    direction = math.degrees(math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0.0
    
    return speed, direction


def point_in_polygon(point: Tuple[float, float], polygon: List[float]) -> bool:
    """Kiểm tra điểm có trong polygon không (simplified: rectangle)"""
    x, y = point
    return polygon[0] <= x <= polygon[2] and polygon[1] <= y <= polygon[3]


# ============================================================
# CORE MODULES
# ============================================================

class FilterObjects(Node):
    """Lọc tracked objects theo class_name"""
    
    def __init__(self, labels: List[str], name: str = "FilterObjects"):
        super().__init__(name)
        self.labels = labels
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        ctx.tracked_objects = [
            obj for obj in ctx.tracked_objects 
            if obj.obj_name in self.labels
        ]
        print(f"[{self.name}] Filtered to {len(ctx.tracked_objects)} objects with labels {self.labels}")
        return ctx


class CheckZone(Node):
    """Kiểm tra object có trong zone không"""
    
    def __init__(self, zones: List[ZoneArea], condition: ZoneCondition = ZoneCondition.IS_INSIDE, 
                 name: str = "CheckZone"):
        super().__init__(name)
        self.zones = zones
        self.condition = condition
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        filtered = []
        
        for obj in ctx.tracked_objects:
            if not obj.predictions:
                continue
            
            last_pred = obj.predictions[-1]
            if not last_pred.detection:
                continue
            
            bbox = last_pred.detection.bbox
            center = get_bbox_center(bbox)
            
            in_any_zone = False
            for zone in self.zones:
                if point_in_polygon(center, zone.polygon):
                    in_any_zone = True
                    break
            
            if self.condition == ZoneCondition.IS_INSIDE and in_any_zone:
                filtered.append(obj)
            elif self.condition == ZoneCondition.IS_OUTSIDE and not in_any_zone:
                filtered.append(obj)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] {len(ctx.tracked_objects)} objects meet zone condition")
        return ctx


class CheckDistance(Node):
    """Kiểm tra khoảng cách giữa các objects và lưu thông tin tương tác"""
    
    def __init__(self, max_distance: float, labels: Optional[List[str]] = None, 
                 name: str = "CheckDistance"):
        super().__init__(name)
        self.max_distance = max_distance
        self.labels = labels
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        filtered = []
        objects = ctx.tracked_objects
        
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                # Check label condition
                if self.labels:
                    if len(self.labels) == 2:
                        valid_pair = (
                            (obj1.obj_name == self.labels[0] and obj2.obj_name == self.labels[1]) or
                            (obj1.obj_name == self.labels[1] and obj2.obj_name == self.labels[0])
                        )
                        if not valid_pair:
                            continue
                    elif len(self.labels) == 1:
                        if obj1.obj_name != self.labels[0] or obj2.obj_name != self.labels[0]:
                            continue
                
                # Calculate distance
                if obj1.predictions and obj2.predictions:
                    bbox1 = obj1.predictions[-1].detection.bbox
                    bbox2 = obj2.predictions[-1].detection.bbox
                    
                    distance = calculate_distance(bbox1, bbox2)
                    
                    if distance <= self.max_distance:
                        # Store nearby info
                        if not obj1.current_attribute.nearby_objects:
                            obj1.current_attribute.nearby_objects = {}
                        obj1.current_attribute.nearby_objects[obj2.track_id] = distance
                        
                        if not obj2.current_attribute.nearby_objects:
                            obj2.current_attribute.nearby_objects = {}
                        obj2.current_attribute.nearby_objects[obj1.track_id] = distance
                        
                        if obj1 not in filtered:
                            filtered.append(obj1)
                        if obj2 not in filtered:
                            filtered.append(obj2)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] Found {len(ctx.tracked_objects)} objects within {self.max_distance}m")
        return ctx


class CheckPPE(Node):
    """Kiểm tra trang bị bảo hộ"""
    
    def __init__(self, ppe_type: PPEType, condition: PPECondition = PPECondition.IS_WEARING,
                 name: str = "CheckPPE"):
        super().__init__(name)
        self.ppe_type = ppe_type
        self.condition = condition
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        filtered = []
        
        for obj in ctx.tracked_objects:
            if obj.obj_name != "person":
                continue
            
            ppe_status = obj.current_attribute.ppe_status or {}
            is_wearing = ppe_status.get(self.ppe_type.value, False)
            
            if self.condition == PPECondition.IS_WEARING and is_wearing:
                filtered.append(obj)
            elif self.condition == PPECondition.IS_NOT_WEARING and not is_wearing:
                filtered.append(obj)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] {len(ctx.tracked_objects)} persons meet PPE condition")
        return ctx


class CheckUseItem(Node):
    """Kiểm tra việc sử dụng vật phẩm"""
    
    def __init__(self, item: ItemType, name: str = "CheckUseItem"):
        super().__init__(name)
        self.item = item
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        filtered = []
        
        for obj in ctx.tracked_objects:
            if obj.obj_name != "person":
                continue
            
            # Check if person is using the item (simplified: check action)
            if obj.current_attribute.action == f"using_{self.item.value}":
                filtered.append(obj)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] {len(ctx.tracked_objects)} persons using {self.item.value}")
        return ctx


class CheckPose(Node):
    """Kiểm tra tư thế của người"""
    
    def __init__(self, pose_condition: PoseCondition, name: str = "CheckPose"):
        super().__init__(name)
        self.pose_condition = pose_condition
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        filtered = []
        
        for obj in ctx.tracked_objects:
            # if obj.obj_name != "person":
            #     continue
            
            if not obj.predictions or not obj.predictions[-1].pose:
                continue
            
            # Simplified: check action attribute
            if obj.current_attribute.action == self.pose_condition.value:
                filtered.append(obj)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] {len(ctx.tracked_objects)} persons in {self.pose_condition.value} pose")
        return ctx


class CheckMoving(Node):
    """Kiểm tra trạng thái di chuyển"""
    
    def __init__(self, threshold: float, condition: MovingCondition, name: str = "CheckMoving"):
        super().__init__(name)
        self.threshold = threshold
        self.condition = condition
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx

        filtered = []
        
        for obj in ctx.tracked_objects:
            speed, _ = calculate_velocity(obj)
            
            is_moving = speed > self.threshold
            
            if self.condition == MovingCondition.MOVING and is_moving:
                filtered.append(obj)
            elif self.condition == MovingCondition.STATIONARY and not is_moving:
                filtered.append(obj)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] {len(ctx.tracked_objects)} objects are {self.condition.value}")
        return ctx


class AnalyzingSpeed(Node):
    """Tính toán và phân tích vận tốc"""
    
    def __init__(self, condition: SpeedCondition, value: float, time_window: float = 3.0,
                 name: str = "AnalyzingSpeed"):
        super().__init__(name)
        self.condition = condition
        self.value = value
        self.time_window = time_window
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx

        filtered = []
        
        for obj in ctx.tracked_objects:
            speed, direction = calculate_velocity(obj, self.time_window)
            
            # Update current attribute
            obj.current_attribute.speed = speed
            obj.current_attribute.direction = direction
            
            # Check condition
            if self.condition == SpeedCondition.IS_FASTER_THAN and speed > self.value:
                filtered.append(obj)
            elif self.condition == SpeedCondition.IS_SLOWER_THAN and speed < self.value:
                filtered.append(obj)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] {len(ctx.tracked_objects)} objects meet speed condition")
        return ctx


class CheckState(Node):
    """Kiểm tra trạng thái của object"""
    
    def __init__(self, state: str, name: str = "CheckState"):
        super().__init__(name)
        self.state = state
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx

        ctx.tracked_objects = [
            obj for obj in ctx.tracked_objects
            if obj.current_attribute.state == self.state
        ]
        print(f"[{self.name}] {len(ctx.tracked_objects)} objects in state '{self.state}'")
        return ctx


class CheckAltitude(Node):
    """Kiểm tra độ cao của object"""
    
    def __init__(self, ref_surface: ZoneArea, condition: AltitudeCondition, 
                 threshold: float = 0.0, name: str = "CheckAltitude"):
        super().__init__(name)
        self.ref_surface = ref_surface
        self.condition = condition
        self.threshold = threshold
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx

        filtered = []
        
        for obj in ctx.tracked_objects:
            if not obj.predictions:
                continue
            
            bbox = obj.predictions[-1].detection.bbox
            obj_bottom = bbox[3]  # y2
            ref_height = self.ref_surface.polygon[3]  # reference surface y
            
            altitude = ref_height - obj_bottom
            obj.current_attribute.altitude = altitude
            
            if self.condition == AltitudeCondition.IS_HIGHER and altitude > self.threshold:
                filtered.append(obj)
            elif self.condition == AltitudeCondition.IS_LOWER and altitude < self.threshold:
                filtered.append(obj)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] {len(ctx.tracked_objects)} objects meet altitude condition")
        return ctx


class CheckGroup(Node):
    """Nhóm các objects dựa trên khoảng cách"""
    
    def __init__(self, min_group_size: int, max_group_size: int, max_distance: float,
                 labels: Optional[List[str]] = None, name: str = "CheckGroup"):
        super().__init__(name)
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.max_distance = max_distance
        self.labels = labels
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        objects = ctx.tracked_objects
        if self.labels:
            objects = [obj for obj in objects if obj.obj_name in self.labels]
        
        # Simple clustering by distance
        groups = []
        used = set()
        
        for i, obj1 in enumerate(objects):
            if i in used:
                continue
            
            group = [obj1]
            used.add(i)
            
            for j, obj2 in enumerate(objects[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if obj2 is close to any member in current group
                for group_member in group:
                    if group_member.predictions and obj2.predictions:
                        bbox1 = group_member.predictions[-1].detection.bbox
                        bbox2 = obj2.predictions[-1].detection.bbox
                        distance = calculate_distance(bbox1, bbox2)
                        
                        if distance <= self.max_distance:
                            group.append(obj2)
                            used.add(j)
                            break
            
            if self.min_group_size <= len(group) <= self.max_group_size:
                groups.append(group)
        
        # Assign group IDs
        filtered = []
        for group_id, group in enumerate(groups):
            for obj in group:
                obj.current_attribute.group_id = group_id
                filtered.append(obj)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] Found {len(groups)} groups with {len(filtered)} objects")
        return ctx


class CheckCoMovement(Node):
    """Kiểm tra di chuyển cùng nhau"""
    
    def __init__(self, time_window: float, velocity_tolerance: float, 
                 direction_tolerance: float, labels: Optional[List[str]] = None,
                 name: str = "CheckCoMovement"):
        super().__init__(name)
        self.time_window = time_window
        self.velocity_tolerance = velocity_tolerance
        self.direction_tolerance = direction_tolerance
        self.labels = labels
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        objects = ctx.tracked_objects
        filtered = []
        
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                # Check label condition
                if self.labels and len(self.labels) == 2:
                    valid_pair = (
                        (obj1.obj_name == self.labels[0] and obj2.obj_name == self.labels[1]) or
                        (obj1.obj_name == self.labels[1] and obj2.obj_name == self.labels[0])
                    )
                    if not valid_pair:
                        continue
                
                # Calculate velocities
                speed1, dir1 = calculate_velocity(obj1, self.time_window)
                speed2, dir2 = calculate_velocity(obj2, self.time_window)
                
                velocity_diff = abs(speed1 - speed2)
                direction_diff = abs(dir1 - dir2)
                
                # Normalize direction difference to [0, 180]
                if direction_diff > 180:
                    direction_diff = 360 - direction_diff
                
                # Check if moving together
                if (velocity_diff <= self.velocity_tolerance and 
                    direction_diff <= self.direction_tolerance):
                    
                    # Store comovement info
                    if not obj1.current_attribute.comovement:
                        obj1.current_attribute.comovement = {}
                    obj1.current_attribute.comovement[obj2.track_id] = (velocity_diff, direction_diff)
                    
                    if not obj2.current_attribute.comovement:
                        obj2.current_attribute.comovement = {}
                    obj2.current_attribute.comovement[obj1.track_id] = (velocity_diff, direction_diff)
                    
                    if obj1 not in filtered:
                        filtered.append(obj1)
                    if obj2 not in filtered:
                        filtered.append(obj2)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] Found {len(ctx.tracked_objects)} objects moving together")
        return ctx


class CheckAbruptEvent(Node):
    """Phát hiện thay đổi độ cao đột ngột"""
    
    def __init__(self, vertical_displacement_threshold: float, time_window: float,
                 name: str = "CheckAbruptEvent"):
        super().__init__(name)
        self.threshold = vertical_displacement_threshold
        self.time_window = time_window
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        filtered = []
        
        for obj in ctx.tracked_objects:
            recent_preds = [p for p in obj.predictions 
                          if p.timestamp >= ctx.timestamp - self.time_window * 1000]
            
            if len(recent_preds) < 2:
                continue
            
            y_positions = [p.detection.bbox[3] for p in recent_preds]  # bottom of bbox
            max_displacement = max(y_positions) - min(y_positions)
            
            if max_displacement > self.threshold:
                filtered.append(obj)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] Detected {len(ctx.tracked_objects)} abrupt events")
        return ctx


class CheckImpact(Node):
    """Phát hiện va chạm"""
    
    def __init__(self, impact_sensitivity: ImpactSensitivity, 
                 labels: Optional[List[str]] = None, name: str = "CheckImpact"):
        super().__init__(name)
        self.sensitivity = impact_sensitivity
        self.labels = labels
        
        # Define thresholds based on sensitivity
        self.thresholds = {
            ImpactSensitivity.HIGH: {"distance": 0.5, "speed": 1.0},
            ImpactSensitivity.MEDIUM: {"distance": 1.0, "speed": 2.0},
            ImpactSensitivity.LOW: {"distance": 2.0, "speed": 5.0}
        }
    
    def process(self, ctx: Context) -> Context:
        objects = ctx.tracked_objects
        filtered = []
        threshold = self.thresholds[self.sensitivity]
        
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                # Check labels
                if self.labels:
                    if obj1.obj_name not in self.labels or obj2.obj_name not in self.labels:
                        continue
                
                # Calculate distance and speeds
                if obj1.predictions and obj2.predictions:
                    bbox1 = obj1.predictions[-1].detection.bbox
                    bbox2 = obj2.predictions[-1].detection.bbox
                    distance = calculate_distance(bbox1, bbox2)
                    
                    speed1, _ = calculate_velocity(obj1)
                    speed2, _ = calculate_velocity(obj2)
                    
                    # Check for impact
                    if (distance <= threshold["distance"] and 
                        (speed1 >= threshold["speed"] or speed2 >= threshold["speed"])):
                        
                        if obj1 not in filtered:
                            filtered.append(obj1)
                        if obj2 not in filtered:
                            filtered.append(obj2)
        
        ctx.tracked_objects = filtered
        print(f"[{self.name}] Detected {len(ctx.tracked_objects)} potential impacts")
        return ctx


# ============================================================
# UPDATING MODULES
# ============================================================

class Timer(Node):
    """Khởi tạo/cập nhật timer cho objects"""
    
    def __init__(self, timer_name: str, duration: float, name: str = "Timer"):
        super().__init__(name)
        self.timer_name = timer_name
        self.duration = duration
        self.cached_durations: Dict[int, float] = {self.timer_name: {}}
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        for obj in ctx.tracked_objects:
            if obj.track_id not in self.cached_durations[self.timer_name]:
                # Start timer
                self.cached_durations[self.timer_name][obj.track_id] = 0
            else:
                # Update timer (assume 1 frame = 1 second for simplicity)
                self.cached_durations[self.timer_name][obj.track_id] += 1
        
        # Filter objects that exceed duration
        ctx.tracked_objects = [
            obj for obj in ctx.tracked_objects
            if self.cached_durations.get(self.timer_name, {}).get(obj.track_id, 0) >= self.duration
        ]
        
        # print(f"[{self.name}] {len(ctx.tracked_objects)} objects exceed {self.duration}s")
        return ctx


class StopTimer(Node):
    """Dừng timer"""
    
    def __init__(self, timer_name: str, name: str = "StopTimer"):
        super().__init__(name)
        self.timer_name = timer_name
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        for obj in ctx.tracked_objects:
            if self.timer_name in obj.current_attribute.timers:
                final_time = obj.current_attribute.timers[self.timer_name]
                obj.current_attribute.timers[f"{self.timer_name}_stopped"] = final_time
                del obj.current_attribute.timers[self.timer_name]
        
        print(f"[{self.name}] Stopped timer '{self.timer_name}' for {len(ctx.tracked_objects)} objects")
        return ctx


class UpdateVariable(Node):
    """Cập nhật biến toàn cục"""
    
    def __init__(self, var_name: str, operation: str = "set", value: Any = None, 
                 name: str = "UpdateVariable"):
        super().__init__(name)
        self.var_name = var_name
        self.operation = operation
        self.value = value
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        if self.operation == "set":
            ctx.global_vars[self.var_name] = self.value
        elif self.operation == "increment":
            ctx.global_vars[self.var_name] = ctx.global_vars.get(self.var_name, 0) + 1
        elif self.operation == "decrement":
            ctx.global_vars[self.var_name] = ctx.global_vars.get(self.var_name, 0) - 1
        elif self.operation == "count":
            ctx.global_vars[self.var_name] = len(ctx.tracked_objects)
        elif self.operation == "append":
            if self.var_name not in ctx.global_vars:
                ctx.global_vars[self.var_name] = []
            ctx.global_vars[self.var_name].extend([obj.track_id for obj in ctx.tracked_objects])
        
        print(f"[{self.name}] Updated '{self.var_name}' = {ctx.global_vars.get(self.var_name)}")
        return ctx


class Counter(Node):
    """Đếm số lượng objects"""
    
    def __init__(self, var_name: str, operation: str = "increment", name: str = "Counter"):
        super().__init__(name)
        self.var_name = var_name
        self.operation = operation
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        if self.operation == "increment":
            ctx.global_vars[self.var_name] = ctx.global_vars.get(self.var_name, 0) + len(ctx.tracked_objects)
        elif self.operation == "decrement":
            ctx.global_vars[self.var_name] = ctx.global_vars.get(self.var_name, 0) - len(ctx.tracked_objects)
        elif self.operation == "reset":
            ctx.global_vars[self.var_name] = 0
        
        print(f"[{self.name}] Counter '{self.var_name}' = {ctx.global_vars[self.var_name]}")
        return ctx


class CheckQuantity(Node):
    """Kiểm tra số lượng objects"""
    
    def __init__(self, label_name: str, min_count: Optional[int] = None, 
                 max_count: Optional[int] = None, name: str = "CheckQuantity"):
        super().__init__(name)
        self.label_name = label_name
        self.min_count = min_count
        self.max_count = max_count
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        count = len([obj for obj in ctx.tracked_objects if obj.obj_name == self.label_name])
        ctx.global_vars[f"{self.label_name}_count"] = count
        
        # Keep all objects if quantity check passes
        passes = True
        if self.min_count is not None and count < self.min_count:
            passes = False
        if self.max_count is not None and count > self.max_count:
            passes = False
        
        if not passes:
            ctx.tracked_objects = []
        
        print(f"[{self.name}] {self.label_name} count: {count}, passes: {passes}")
        return ctx


# ============================================================
# CONTROL FLOW MODULES
# ============================================================

class ConditionBranch(Node):
    """Node điều kiện: chia luồng thành true_branch và false_branch"""
    
    def __init__(self, condition: Callable[[Context], bool], name: str = "ConditionBranch"):
        super().__init__(name)
        self.condition = condition
        self.true_branch: Optional[Node] = None
        self.false_branch: Optional[Node] = None
    
    def set_branches(self, true_branch: Node, false_branch: Optional[Node] = None):
        self.true_branch = true_branch
        self.false_branch = false_branch
        return self
    
    def execute(self, ctx: Context) -> List[Context]:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        condition_result = self.condition(ctx)
        print(f"[{self.name}] Condition result: {condition_result}")
        
        results = []
        
        if condition_result:
            if self.true_branch:
                results.extend(self.true_branch.execute(ctx))
            else:
                results.append(ctx)
        else:
            if self.false_branch:
                results.extend(self.false_branch.execute(ctx))
            else:
                results.append(ctx)
        
        return results


class MergeNode(Node):
    """Node merge: gộp nhiều luồng lại"""
    
    def __init__(self, expected_inputs: int = 2, name: str = "MergeNode"):
        super().__init__(name)
        self.expected_inputs = expected_inputs
        self._contexts: List[Context] = []
        self._received_sources: set = set()
    
    def add_context(self, ctx: Context, source_name: str = None):
        """
        Add context from a branch.
        - source_name: optional label (like the upstream node name)
        """
        self._contexts.append(ctx)
        if source_name:
            self._received_sources.add(source_name)
        else:
            # fallback: count anonymous sources
            self._received_sources.add(len(self._contexts))
            
    def process(self, ctx: Context) -> Context:
        """
        Called each time an upstream node sends context here.
        Waits until all expected inputs arrive before merging.
        """
        # if ctx == None or ctx.tracked_objects == []:
        #     print(f"[{self.name}] Empty context")
        #     return ctx
        # add incoming context with its source name
        source_name = getattr(ctx, "from_node", None)
        self.add_context(ctx, source_name)

        if len(self._received_sources) < self.expected_inputs:
            print(f"[{self.name}] Waiting... ({len(self._received_sources)}/{self.expected_inputs})")
            return ctx

        # all inputs ready -> merge
        all_objects = []
        seen_ids = set()
        for c in self._contexts:
            for obj in c.tracked_objects:
                if obj.track_id not in seen_ids:
                    all_objects.append(obj)
                    seen_ids.add(obj.track_id)

        merged_ctx = deepcopy(self._contexts[-1])
        merged_ctx.tracked_objects = all_objects
        self._contexts.clear()
        self._received_sources.clear()

        print(f"[{self.name}]  Merged {len(all_objects)} objects from {self.expected_inputs} inputs")
        return merged_ctx


class LogicCode(Node):
    """Thực thi custom logic code"""
    
    def __init__(self, logic_fn: Callable[[Context], Context], name: str = "LogicCode") -> Context:
        super().__init__(name)
        self.logic_fn = logic_fn
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        result = self.logic_fn(ctx)
        print(f"[{self.name}] Custom logic executed")
        return result

class Validate(Node):
    """Thực thi custom logic code"""
    
    def __init__(self, tracked_id_event: Tracked_Id_Event, name="Validate Tracked IDs"):
        super().__init__(name)
        self.tracked_id_event = tracked_id_event
    
    def process(self, ctx: Context) -> Context:     
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.name}] Empty context")
            return ctx
        
        valid_objects = []
        for obj in ctx.tracked_objects:
            if not self.tracked_id_event.check(obj.track_id):
                print(f"[{self.name}] Validated object ID: {obj.track_id}")
                valid_objects.append(obj)
            else:
                print(f"[{self.name}] Skipped object ID: {obj.track_id}")

        ctx.tracked_objects = valid_objects

        return ctx


# ============================================================
# EVENT MODULE
# ============================================================

class EventSender(Node):
    """Tạo và gửi event"""
    
    def __init__(self, event_name: str, include_data: bool = True, name: str = "EventSender", tracked_id_event: Tracked_Id_Event = None):
        super().__init__(name)
        self.event_name = event_name
        self.include_data = include_data
        self.processed_event_tracked_ids = tracked_id_event
    
    def process(self, ctx: Context) -> Context:
        if ctx == None or ctx.tracked_objects == []:
            print(f"[{self.event_name}] No tracked_object to send event in FRAME-{ctx.frame_id}")
            return ctx
        
        if ctx.tracked_objects:
            event_data = {
                "tracked_ids": [obj.track_id for obj in ctx.tracked_objects],
                "count": len(ctx.tracked_objects)
            }
            
            if self.include_data:
                # Add additional information
                event_data["objects"] = []
                for obj in ctx.tracked_objects:
                    obj_info = {
                        "track_id": obj.track_id,
                        "label": obj.obj_name,
                    }
                    
                    if obj.current_attribute.nearby_objects:
                        obj_info["nearby"] = obj.current_attribute.nearby_objects
                    if obj.current_attribute.speed:
                        obj_info["speed"] = obj.current_attribute.speed
                    if obj.current_attribute.group_id is not None:
                        obj_info["group_id"] = obj.current_attribute.group_id
                    
                    event_data["objects"].append(obj_info)
                    
                    if self.processed_event_tracked_ids:
                        self.processed_event_tracked_ids.update(track_id=obj.track_id, event_name=self.event_name)
                    else:
                        pass  # No tracking of processed IDs
            
            event = Event(
                name=self.event_name,
                frame_id=ctx.frame_id,
                timestamp=ctx.timestamp,
                data=event_data
            )
            ctx.events.append(event)
            print(f"[EVENT] {event.name} at frame {event.frame_id}: {len(ctx.tracked_objects)} objects")
        
        return ctx


# ============================================================
# DEMO SCENARIOS
# ============================================================

def create_sample_context() -> Context:
    """Tạo context mẫu với nhiều tracked objects"""
    return Context(
        frame_id=1,
        timestamp=1000,
        tracked_objects=[
            TrackedObject(
                track_id=1,
                obj_name="person",
                current_attribute=Attribute(
                    class_name="person",
                    ppe_status={"helmet": False, "vest": True}
                ),
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[10, 20, 30, 40], score=0.9, class_name="person"))
                ]
            ),
            TrackedObject(
                track_id=2,
                obj_name="person",
                current_attribute=Attribute(
                    class_name="person",
                    ppe_status={"helmet": True, "vest": True},
                    action="raise_hand"
                ),
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[50, 60, 70, 80], score=0.85, class_name="person"))
                ]
            ),
            TrackedObject(
                track_id=3,
                obj_name="staff",
                current_attribute=Attribute(class_name="staff"),
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[100, 120, 120, 140], score=0.88, class_name="staff"))
                ]
            ),
            TrackedObject(
                track_id=4,
                obj_name="MHE",
                current_attribute=Attribute(class_name="MHE"),
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[15, 25, 35, 45], score=0.92, class_name="MHE"))
                ]
            ),
            TrackedObject(
                track_id=5,
                obj_name="vehicle",
                current_attribute=Attribute(class_name="vehicle"),
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[200, 200, 250, 250], score=0.87, class_name="vehicle"))
                ]
            )
        ]
    )


def demo_ppe_violation():
    """
    Demo: Phát hiện người không đeo mũ bảo hiểm
    """
    print("\n" + "="*60)
    print("DEMO 1: PPE VIOLATION - Không đeo mũ bảo hiểm")
    print("="*60)
    
    # Pipeline
    filter_person = FilterObjects(labels=["person"], name="FilterPerson")
    check_ppe = CheckPPE(ppe_type=PPEType.HELMET, 
                         condition=PPECondition.IS_NOT_WEARING,
                         name="CheckHelmet")
    event_sender = EventSender(event_name="PPE_VIOLATION", name="SendPPEAlert")
    
    # Connect
    filter_person.add_next(check_ppe)
    check_ppe.add_next(event_sender)
    
    # Execute
    ctx = create_sample_context()
    results = filter_person.execute(ctx)
    
    print(f"\n Total events: {len(results[0].events)}")
    for event in results[0].events:
        print(f"  - {event.name}: {event.data}")


def demo_proximity_alert():
    """
    Demo: Cảnh báo người quá gần MHE
    """
    print("\n" + "="*60)
    print("DEMO 2: PROXIMITY ALERT - Người quá gần xe MHE")
    print("="*60)
    
    # Pipeline
    filter_objects = FilterObjects(labels=["person", "MHE"], name="FilterPersonMHE")
    check_distance = CheckDistance(max_distance=50, labels=["person", "MHE"], 
                                   name="CheckProximity")
    event_sender = EventSender(event_name="PROXIMITY_WARNING", name="SendProximityAlert")
    
    # Connect
    filter_objects.add_next(check_distance)
    check_distance.add_next(event_sender)
    
    # Execute
    ctx = create_sample_context()
    results = filter_objects.execute(ctx)
    
    print(f"\n Total events: {len(results[0].events)}")
    for event in results[0].events:
        print(f"  - {event.name}: {event.data}")


def demo_group_detection():
    """
    Demo: Phát hiện nhóm người tụ tập
    """
    print("\n" + "="*60)
    print("DEMO 3: GROUP DETECTION - Phát hiện nhóm người")
    print("="*60)
    
    # Add more people for grouping
    ctx = create_sample_context()
    ctx.tracked_objects.extend([
        TrackedObject(
            track_id=6,
            obj_name="person",
            predictions=[
                Prediction(frame_id=1, timestamp=1000,
                          detection=Detection(bbox=[12, 22, 32, 42], score=0.9, class_name="person"))
            ]
        ),
        TrackedObject(
            track_id=7,
            obj_name="person",
            predictions=[
                Prediction(frame_id=1, timestamp=1000,
                          detection=Detection(bbox=[14, 24, 34, 44], score=0.9, class_name="person"))
            ]
        )
    ])
    
    # Pipeline
    filter_person = FilterObjects(labels=["person"], name="FilterPerson")
    check_group = CheckGroup(min_group_size=2, max_group_size=5, max_distance=30,
                            labels=["person"], name="DetectGroups")
    event_sender = EventSender(event_name="GROUP_DETECTED", name="SendGroupAlert")
    
    # Connect
    filter_person.add_next(check_group)
    check_group.add_next(event_sender)
    
    # Execute
    results = filter_person.execute(ctx)
    
    print(f"\n Total events: {len(results[0].events)}")
    for event in results[0].events:
        print(f"  - {event.name}: {event.data}")


def demo_comovement_detection():
    """
    Demo: Phát hiện người và xe di chuyển cùng nhau
    """
    print("\n" + "="*60)
    print("DEMO 4: CO-MOVEMENT - Người và xe di chuyển cùng nhau")
    print("="*60)
    
    # Create context with movement history
    ctx = Context(
        frame_id=5,
        timestamp=5000,
        tracked_objects=[
            TrackedObject(
                track_id=1,
                obj_name="person",
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[10, 20, 30, 40], score=0.9)),
                    Prediction(frame_id=3, timestamp=3000,
                              detection=Detection(bbox=[40, 50, 60, 70], score=0.9)),
                    Prediction(frame_id=5, timestamp=5000,
                              detection=Detection(bbox=[70, 80, 90, 100], score=0.9))
                ]
            ),
            TrackedObject(
                track_id=2,
                obj_name="vehicle",
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[15, 25, 35, 45], score=0.9)),
                    Prediction(frame_id=3, timestamp=3000,
                              detection=Detection(bbox=[45, 55, 65, 75], score=0.9)),
                    Prediction(frame_id=5, timestamp=5000,
                              detection=Detection(bbox=[75, 85, 95, 105], score=0.9))
                ]
            )
        ]
    )
    
    # Pipeline
    filter_objects = FilterObjects(labels=["person", "vehicle"], name="FilterPersonVehicle")
    check_comovement = CheckCoMovement(time_window=3.0, velocity_tolerance=5.0,
                                       direction_tolerance=30.0, labels=["person", "vehicle"],
                                       name="DetectCoMovement")
    event_sender = EventSender(event_name="COMOVEMENT_DETECTED", name="SendCoMovementAlert")
    
    # Connect
    filter_objects.add_next(check_comovement)
    check_comovement.add_next(event_sender)
    
    # Execute
    results = filter_objects.execute(ctx)
    
    print(f"\n Total events: {len(results[0].events)}")
    for event in results[0].events:
        print(f"  - {event.name}: {event.data}")


def demo_zone_timer():
    """
    Demo: Đếm thời gian người ở trong zone nguy hiểm
    """
    print("\n" + "="*60)
    print("DEMO 5: ZONE TIMER - Ở quá lâu trong vùng nguy hiểm")
    print("="*60)
    
    # Simulate multiple frames
    ctx1 = create_sample_context()
    ctx1.frame_id = 1
    
    # Pipeline
    filter_person = FilterObjects(labels=["person"], name="FilterPerson")
    check_zone = CheckZone(zones=[ZoneArea(zone_id="danger_zone", polygon=[0, 0, 100, 100])],
                          condition=ZoneCondition.IS_INSIDE, name="CheckDangerZone")
    timer = Timer(timer_name="in_danger_zone", duration=3, name="TimerDangerZone")
    event_sender = EventSender(event_name="ZONE_TIMEOUT", name="SendZoneAlert")
    
    # Connect
    filter_person.add_next(check_zone)
    check_zone.add_next(timer)
    timer.add_next(event_sender)
    
    # Simulate 5 frames
    results = None
    for frame_id in range(1, 6):
        ctx = create_sample_context()
        ctx.frame_id = frame_id
        ctx.timestamp = frame_id * 1000
        
        print(f"\n--- Frame {frame_id} ---")
        results = filter_person.execute(ctx)
    
    print(f"\n Total events: {len(results[0].events) if results else 0}")
    if results:
        for event in results[0].events:
            print(f"  - {event.name}: {event.data}")


def demo_conditional_branch():
    """
    Demo: Điều kiện phân nhánh - nếu có customer giơ tay mà không có staff gần thì cảnh báo
    """
    print("\n" + "="*60)
    print("DEMO 6: CONDITIONAL BRANCH - Customer giơ tay không có staff")
    print("="*60)
    
    # Pipeline cho customer
    filter_customer = FilterObjects(labels=["person"], name="FilterCustomer")
    check_pose = CheckPose(pose_condition=PoseCondition.RAISE_HAND, name="CheckRaiseHand")
    
    # Pipeline cho staff
    filter_staff = FilterObjects(labels=["staff"], name="FilterStaff")
    
    # Merge để check distance
    check_distance = CheckDistance(max_distance=100, labels=["person", "staff"],
                                   name="CheckStaffNearby")
    
    # Condition: Nếu không có staff gần (tracked_objects rỗng sau CheckDistance)
    condition = ConditionBranch(
        condition=lambda ctx: len(ctx.tracked_objects) == 0,
        name="CheckNoStaffNearby"
    )
    
    # True branch: Không có staff → cảnh báo
    timer = Timer(timer_name="waiting_service", duration=2, name="WaitTimer")
    event_sender = EventSender(event_name="NO_SERVICE", name="SendNoServiceAlert")
    
    # Connect
    filter_customer.add_next(check_pose)
    check_pose.add_next(check_distance)
    check_distance.add_next(condition)
    condition.set_branches(true_branch=timer, false_branch=None)
    timer.add_next(event_sender)
    
    # Test với context không có staff gần
    ctx = create_sample_context()
    # Move staff far away
    ctx.tracked_objects[2].predictions[0].detection.bbox = [500, 500, 520, 520]
    
    # Execute multiple frames để trigger timer
    results = None
    for frame_id in range(1, 4):
        ctx_frame = create_sample_context()
        ctx_frame.frame_id = frame_id
        ctx_frame.tracked_objects[2].predictions[0].detection.bbox = [500, 500, 520, 520]
        
        print(f"\n--- Frame {frame_id} ---")
        results = filter_customer.execute(ctx_frame)
    
    print(f"\n Total events: {len(results[0].events) if results else 0}")
    if results:
        for event in results[0].events:
            print(f"  - {event.name}: {event.data}")


def demo_impact_detection():
    """
    Demo: Phát hiện va chạm với độ nhạy khác nhau
    """
    print("\n" + "="*60)
    print("DEMO 7: IMPACT DETECTION - Phát hiện va chạm")
    print("="*60)
    
    # Create context with fast moving objects close together
    ctx = Context(
        frame_id=3,
        timestamp=3000,
        tracked_objects=[
            TrackedObject(
                track_id=1,
                obj_name="person",
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[10, 20, 30, 40], score=0.9)),
                    Prediction(frame_id=2, timestamp=2000,
                              detection=Detection(bbox=[20, 30, 40, 50], score=0.9)),
                    Prediction(frame_id=3, timestamp=3000,
                              detection=Detection(bbox=[30, 40, 50, 60], score=0.9))
                ]
            ),
            TrackedObject(
                track_id=2,
                obj_name="MHE",
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[50, 60, 70, 80], score=0.9)),
                    Prediction(frame_id=2, timestamp=2000,
                              detection=Detection(bbox=[40, 50, 60, 70], score=0.9)),
                    Prediction(frame_id=3, timestamp=3000,
                              detection=Detection(bbox=[31, 41, 51, 61], score=0.9))
                ]
            )
        ]
    )
    
    # Pipeline
    filter_objects = FilterObjects(labels=["person", "MHE"], name="FilterObjects")
    check_impact = CheckImpact(impact_sensitivity=ImpactSensitivity.MEDIUM,
                               labels=["person", "MHE"], name="DetectImpact")
    event_sender = EventSender(event_name="IMPACT_DETECTED", name="SendImpactAlert")
    
    # Connect
    filter_objects.add_next(check_impact)
    check_impact.add_next(event_sender)
    
    # Execute
    results = filter_objects.execute(ctx)
    
    print(f"\n Total events: {len(results[0].events)}")
    for event in results[0].events:
        print(f"  - {event.name}: {event.data}")


def demo_complex_workflow():
    """
    Demo: Workflow phức tạp - Kết hợp nhiều điều kiện
    Scenario: Phát hiện người không đeo PPE đang ở gần MHE và di chuyển
    """
    print("\n" + "="*60)
    print("DEMO 8: COMPLEX WORKFLOW - Nhiều điều kiện kết hợp")
    print("="*60)
    
    # Create context with moving person without helmet near MHE
    ctx = Context(
        frame_id=2,
        timestamp=2000,
        tracked_objects=[
            TrackedObject(
                track_id=1,
                obj_name="person",
                current_attribute=Attribute(
                    class_name="person",
                    ppe_status={"helmet": False, "vest": True}
                ),
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[10, 20, 30, 40], score=0.9)),
                    Prediction(frame_id=2, timestamp=2000,
                              detection=Detection(bbox=[25, 35, 45, 55], score=0.9))
                ]
            ),
            TrackedObject(
                track_id=2,
                obj_name="MHE",
                predictions=[
                    Prediction(frame_id=1, timestamp=1000,
                              detection=Detection(bbox=[15, 25, 35, 45], score=0.9)),
                    Prediction(frame_id=2, timestamp=2000,
                              detection=Detection(bbox=[15, 25, 35, 45], score=0.9))
                ]
            )
        ]
    )
    
    # Pipeline
    filter_person = FilterObjects(labels=["person"], name="1.FilterPerson")
    check_ppe = CheckPPE(ppe_type=PPEType.HELMET, 
                        condition=PPECondition.IS_NOT_WEARING, name="2.CheckNoHelmet")
    check_moving = CheckMoving(threshold=1.0, condition=MovingCondition.MOVING,
                              name="3.CheckMoving")
    
    # Add MHE to check proximity
    filter_with_mhe = LogicCode(
        logic_fn=lambda ctx: Context(
            frame_id=ctx.frame_id,
            timestamp=ctx.timestamp,
            tracked_objects=ctx.tracked_objects + [
                obj for obj in create_sample_context().tracked_objects 
                if obj.obj_name == "MHE"
            ],
            global_vars=ctx.global_vars,
            events=ctx.events
        ),
        name="4.AddMHEObjects"
    )
    
    check_distance = CheckDistance(max_distance=50, labels=["person", "MHE"],
                                   name="5.CheckNearMHE")
    
    # Count violations
    counter = Counter(var_name="safety_violations", operation="increment",
                     name="6.CountViolations")
    
    event_sender = EventSender(event_name="SAFETY_VIOLATION", name="7.SendAlert")
    
    # Connect pipeline
    filter_person.add_next(check_ppe)
    check_ppe.add_next(check_moving)
    check_moving.add_next(filter_with_mhe)
    filter_with_mhe.add_next(check_distance)
    check_distance.add_next(counter)
    counter.add_next(event_sender)
    
    # Execute
    results = filter_person.execute(ctx)
    
    print(f"\n Total events: {len(results[0].events)}")
    print(f" Safety violations count: {results[0].global_vars.get('safety_violations', 0)}")
    for event in results[0].events:
        print(f"  - {event.name}: {event.data}")


# ============================================================
# MAIN - Run all demos
# ============================================================

if __name__ == "__main__":
    # print("\n" + "="*60)
    # print("LOW-CODE CV RULE SYSTEM - COMPREHENSIVE DEMOS")
    # print("="*60)
    
    # demo_ppe_violation()
    # demo_proximity_alert()
    # demo_group_detection()
    # demo_comovement_detection()
    # demo_zone_timer()
    # demo_conditional_branch()
    # demo_impact_detection()
    demo_complex_workflow()
    
    # print("\n" + "="*60)
    # print("ALL DEMOS COMPLETED!")
    # print("="*60)