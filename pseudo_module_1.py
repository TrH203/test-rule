from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


# ============================================================
# ! BASIC DATA TYPES
# ============================================================

@dataclass
class Detection:
    bbox: List[float]                     # [x1, y1, x2, y2]
    score: float = 0.0
    class_name: Optional[str] = None


@dataclass
class Pose:
    keypoints: List[List[float]]          # (n, 3): [x, y, conf]
    score: float = 0.0


@dataclass
class Segmentation:
    points: List[float]                   # flat or polygon points
    score: float = 0.0


@dataclass
class Attribute:
    class_name: Optional[str] = None
    group: Optional[int] = None
    speed: Optional[float] = None
    direction: Optional[float] = None
    state: Optional[str] = None
    action: Optional[str] = None
    color: Optional[str] = None
    nearby_objects: Optional[Dict] = None
    comovement : Optional[Dict] = None
    time_counter: Optional[function]
    time_trigger: Optional[dict] = {"start": None, "end": None}
    

@dataclass
class ZoneArea:
    zone_id = str
    polygon = List[int]

class PPEType(Enum):
    HELMET = "helmet"
    VEST = "vest"
    SAFETY_SHOES = "safety_shoes"
    
class PPECondition(Enum):
    IS_WEARING = "is_wearing"
    IS_NOT_WEARING = "is_not_wearing"

# ============================================================
# ! PREDICTION & TRACK
# ============================================================
@dataclass
class InputFrame:
    frame_id: int = 0
    timestamp: int = 0
    model_result: List[Dict[str, Any]] 
    """[
        {"tracked_id": 1 "bbox": [...], "score": ..., "class_name": ..., "keypoints": [...], "segmentation": [...]}
        {"tracked_id": 2 "bbox": [...], "score": ..., "class_name": ..., "keypoints": [...], "segmentation": [...]}
    ]
    """

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
    current_attribute: Optional[Attribute] = None
    prezdictions: List[Prediction] = field(default_factory=list)


@dataclass
class Frame:
    frame_id: int
    timestamp: int
    list_tracked_object: List[TrackedObject] = field(default_factory=list)


@dataclass
class Video:
    frames: List[Frame] = field(default_factory=list)

@dataclass
class CodeBlock:
    code: str
    inputs: Any
    outputs: Any
    
# @dataclass
# class PairObject:
#     left: TrackedObject
#     right: TrackedObject


# ============================================================
# ! ENUM AND EVENT
# ============================================================

@dataclass
class Event:
    name: str
    data: Dict[str, Any]

@dataclass
class EventStatus:
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None

class Comparison(Enum):
    LESS_THAN = 1
    GREATER_THAN = 2
    LESS_OR_EQUAL = 3
    GREATER_OR_EQUAL = 4
    EQUAL = 5
    NOT_EQUAL = 6

class Pair(Enum):  
    ONELIST = 1
    TWOLIST = 2

class TimeCondition(Enum):
    LONGER_THAN = "longer_than"
    SHORTER_THAN = "shorter_than"
    
class VariableAction(Enum):
    OVERWRITE = "overwrite"
    APPEND_IF_NOT_EXISTS = "append_if_not_exists"
    INCREMENT = "increment"
    DECREMENT = "decrement"
    DELETE = "delete"


class ZoneCondition(Enum):
    IS_INSIDE = "is_inside"
    IS_OUTSIDE = "is_outside"
    CROSSES_LINE = "crosses_line"

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
    

class AnalysisType(Enum):
    AVG_SPEED = "AVG_SPEED"
    AVG_DECELERATION = "AVG_DECELERATION"
    CURRENT_SPEED = "CURRENT_SPEED"
    CURRENT_DECELERATION = "CURRENT_DECELERATION"

class SpeedCondition(Enum):
    IS_FASTER_THAN = "is_faster_than"
    IS_SLOWER_THAN = "is_slower_than"

class LabelCondition(Enum):
    IS = "is"
    IS_NOT = "is_not"

class AltitudeCondition(Enum):
    IS_HIGHER = 'is_higher'
    IS_LOWER = 'is_under'
    
class CounterAction(Enum):
    INCREASE = 'increase'
    DECREASE = 'decrease'
    RESET = 'reset'
    
class ImpactSensitivity(Enum):
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

# ============================================================
# ! BASE MODULE CLASSES
# ============================================================
class Node:
    def __init__(self):
        pass
    
    def __init__(
        self,
        previous_nodes: Optional[List["Node"]] = None,
        next_nodes: Optional[List["Node"]] = None,
    ):
        self.previous_nodes = previous_nodes or []
        self.next_nodes = next_nodes or []

    def add_next(self, node: "Node"):
        self.next_nodes.append(node)
        node.previous_nodes.append(self)
    
    def process(self, input: Any) -> Any:
        raise NotImplementedError

    def update_cache(self, input: Any) -> Any:
        return input
    
    def update_db(self, input: Any) -> Any:
        return input


class Core_Module(Node):
    """
    Module logic chuẩn, phục vụ công việc cố định
    Nhận vào list tracked object và trả ra list tracked object.
    Khi xử lý bên trong, sẽ lặp qua tất cả các tracked object trong list để xử lý.
    """
    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        return input


class Init_Module(Node):
    """
    Module khởi tạo, module này sẽ không nhận input gì cả.
    Module này sẽ trả ra list tracked object."""
    def process(self) -> List[TrackedObject]:
        return []


class Updating_Module(Node):
    """
    Module cập nhập biến toàn cục, module này sẽ nhận vào 1 tracked object
    và trả ra list tracked object đầu vào để luồng được tiếp tục."""
    def process(self, input: Any) -> Any:
        pass
    
    def update_cache(self, target: str, input: Any) -> Any:
        pass
    
    def update_db(self, target: str, input: Any) -> Any:
        pass


class Event_Module(Node):
    """
    Module sự kiện, module này sẽ nhận vào list tracked object
    và trả ra 1 event. Sau khi trả ra event, module sẽ tự động gửi event đi.
    """
    def process(self, input: List[TrackedObject]) -> None:
        pass
    
    def send_event(self, event: Event) -> "EventStatus":
        print(f"Event sent: {event.name}")
        return EventStatus(success=True, message="Event delivered")
    
    
# ============================================================
# ! IMPLEMENTATION MODULES
# ============================================================

class Trigger_Kafka_Message(Init_Module):
    """Module mô phỏng việc nhận dữ liệu từ Kafka."""
    def process(self) -> List[TrackedObject]:
        pass


class Filter_Objects(Core_Module):
    """
    Lọc tất cả các tracked object trong 1 list label cụ thể
    Sau khi qua module này, chỉ còn tồn tại những tracked object 
    mà sử hữu đúng label đã được định nghĩa trong value.
    """
    def __init__(
        self,
        value: List[str], 
        condition: Comparison
        ) -> None:
        pass
    def process(
        self, 
        input: List[TrackedObject], 
    ) -> List[TrackedObject]:
        pass


# //class Loop_Through_Object(Core_Module):
# //    def process(self, input: List[TrackedObject]) -> TrackedObject:
# //        pass

# class Get_All_Pair(Core_Module):
#     def __init__(
#         self,
#         mode: Pair
#         ) -> None:
#         pass
    
#     def process(
#         self,
#         left: List[TrackedObject],
#         right: List[TrackedObject],
#     ) -> List[Tuple[TrackedObject, TrackedObject]]:
#         pass


class IF(Core_Module):
    """
    Module này nhận vào 1 list tracked object và trả ra 1 list tracked object theo luồng điều kiện.
    Module này sử dùng code giúp ta tạo ra 2 điều kiện
    Module này sẽ trả ra 2 đầu với mỗi đầu là 1 điều kiện
    """
    def __init__(
        self,
        code_block: CodeBlock,
    ):
        pass
    
    def link(self, post_node = List[Node]):
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        pass
    
class Logic_Code(Core_Module):
    """
    Xử lý logic thêm với module code này
    Có thể nhận bất kì input gì và trả ra bất kì output gì.
    """
    def __init__(
        self,
        code_block: CodeBlock,
        context: Optional[Any] = None
    ) -> None:
        pass

    def process(self, input: Any) -> Any:
        pass
    
class Timer(Updating_Module):
    """
    Thực hiện gán thời gian vào trong từng tracked object
    module sẽ gán hàm tính thời gian và điều kiện vào trong 
    tracked object. Hàm sẽ được lưu vào trong
    trackedobject.attribute.time_counter. 
    Hàm này sẽ được cập nhập dựa trên thời gian bắt đầu xuất hiện
    đến khi nó kết thúc sự hiện diện. Ta cũng có 1 điều kiện nếu thỏa điều kiện.
    ta có thể trigger sự kiện = hàm time trên.
    """
    def __init__(
        self,
        timer_name: Optional[str] = None,
        condition: TimeCondition = TimeCondition.LONGER_THAN,
        duration: Optional[float] = 0.0,  # giây
    ) -> Any:
        pass

    def process(self, input: TrackedObject) -> TrackedObject: # ! confuse
        pass
    
    def update_cache(self, target, input):
        return super().update_cache(target, input)
    
    def update_db(self, target, input):
        return super().update_db(target, input)
    
    

class StopTimer(Updating_Module):
    """
    Dừng bộ đếm thời gian đã được khởi tạo trước đó.
    Module này sẽ lưu thời gian vào attribute của tracked object.
    Sử dụng điều kiện (nếu có thể cho qua 1 list các tracked object)
    Module này sẽ nhận vào 1 list tracked object và trả ra 1 list tracked object.
    """
    def __init__(
        self,
        timer_name: Optional[str] = None,
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> Any: # ! confuse
        pass
    
    def update_db(self, target, input):
        return super().update_db(target, input)
    
    def update_cache(self, target, input):
        return super().update_cache(target, input)

class InitUpdateVariable(Updating_Module):
    """
    Define biến 
    """
    def __init__(
        self,
        variable_name: str,
        action: VariableAction = VariableAction.OVERWRITE,
    ) -> None:
        pass

    def process(self, input: Any) -> Any:
        pass
    
    def update_cache(self, target, input):
        return super().update_cache(target, input)
    
    def update_db(self, target, input):
        return super().update_db(target, input)
    

class CheckZone(Core_Module):
    """
    Kiểm tra xem một đối tượng có nằm trong một vùng cụ thể hay không.
    Nhận vào list tracked object và trả ra list tracked object nếu thỏa điều kiện.
    """
    def __init__(
        self,
        zone: List[ZoneArea],
        condition: ZoneCondition = ZoneCondition.IS_INSIDE,
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        """
        Input: TrackedObject
        Output: TrackedObject (nếu điều kiện đúng)
        """
        pass
    
class CheckPPE(Core_Module):
    """
    Kiểm tra trang bị bảo hộ (PPE) của một người dựa trên attributes.ppe_status.
    Nhận vào list tracked object và trả ra list tracked object nếu thỏa điều kiện.
    """

    def __init__(
        self,
        ppe_type: PPEType,
        condition: PPECondition = PPECondition.IS_WEARING,
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        """
        Input: TrackedObject (label 'person')
        Output: TrackedObject (nếu điều kiện đúng)
        """
        pass
    
class CheckUseItem(Core_Module):
    """
    Kiểm tra xem một người có đang sử dụng vật phẩm cụ thể (ví dụ: điện thoại) hay không.
    Nhận vào list tracked object và trả ra list tracked object nếu thỏa điều kiện.
    """

    def __init__(
        self,
        item: ItemType = ItemType.PHONE,
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        """
        Input: TrackedObject (label 'person')
        Output: TrackedObject (nếu điều kiện đúng)
        """
        pass
    
class CheckDistance(Core_Module):
    """
    Kiểm tra khoảng cách giữa các đối tượng trong một danh sách.
    Nhận vào list tracked object và trả ra list tracked object nếu thỏa điều kiện.
    Nếu label = None thì tính và kiểm tra giữa tất cả các đối tượng. (for(for))
    Nếu label = [a, b] thì chỉ tính và kiểm tra giữa các đối tượng có label a với các đối tượng có label b. (for)
    Nếu label = [a] thì chỉ tính và kiểm tra giữa các đối tượng có label a với nhau. (for(for))
    Thông tin sẽ được lưu ở trackedobject1.predictions.[n].attribute.nearby_objects = {trackedobject2.track_id: distance, trackedobject3.track_id: distance, ...}
    """
    def __init__(
        self,
        distance: int,
        condition: Comparison,
        label: Optional[List[str]]
    ) -> None:
        pass
    
    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        pass
    
class CheckPose(Core_Module):
    """
    Kiểm tra tư thế của một người dựa trên keypoints.
    Nhận vào list tracked object và trả ra list tracked object nếu thỏa điều kiện.
    """
    def __init__(
        self,
        pose_condition: PoseCondition,
    ) -> None:
        pass
    
    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        """
        Input: TrackedObject (label 'person')
        Output: TrackedObject (nếu điều kiện đúng)
        """
        pass
    
class CheckMoving(Core_Module):
    """
    Kiểm tra trạng thái di chuyển của một người dựa trên sự thay đổi vị trí
    ở 2 frame cuối dùng để tính vận tốc.
    Nhận vào list tracked object và trả ra list tracked object nếu thỏa điều kiện.
    """
    def __init__(
        self,
        threshold: float,
        condition: MovingCondition,
    ) -> None:
        pass
    
    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        """
        Input: TrackedObject (label 'person')
        Output: TrackedObject (nếu điều kiện đúng)
        """
        pass
    
class AnalyzingSpeed(Core_Module):
    """
    Tính toán toàn bộ sự duy chuyển của từng đối lượng ở trừng frame của nó
    tình toán từ trackedobject.predictions[n].attribute.speed và trackedobject.prediction[n].attribute.direction
    Module này sẽ lưu vận tốc và hướng di chuyển vào trong attribute
    trackedobject.current_attribute.speed và trackedobject.current_attribute.direction
    Module này sẽ nhận vào list tracked object và trả ra list tracked object thỏa điều kiện.
    """

    def __init__(
        self,
        analysis_type: AnalysisType = AnalysisType.AVG_SPEED,
        condition: SpeedCondition = SpeedCondition.IS_FASTER_THAN,
        value: float = 0.0,
        time_window: float = 3.0,  # giây
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        """
        Input: TrackedObject
        Output: TrackedObject (nếu điều kiện đúng)
        """
        pass
    
# ! Confuse
# class CheckLabel(Core_Module):
#     """
#     Kiểm tra nhãn (label) hoặc vai trò (role) của một đối tượng.
#     """

#     def __init__(
#         self,
#         condition: LabelCondition = LabelCondition.IS,
#         value: str = "",
#     ) -> None:
#         pass

#     def process(self, input: TrackedObject) -> TrackedObject: # * Modified
#         """
#         Input: TrackedObject
#         Output: TrackedObject (nếu điều kiện đúng)
#         """
#         pass

class CheckState(Core_Module):
    """
    Kiểm tra trạng thái của một đối tượng (trackedobject.attributes.state).
    """

    def __init__(
        self,
        state: str,
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        """
        Input: TrackedObject
        Output: TrackedObject (nếu điều kiện đúng)
        """
        pass
    
class CheckQuantity(Updating_Module):
    """
    Đếm số lượng đối tượng trong danh sách và lưu vào biến toàn cục.
    Nhận vào tracked object và trả ra số lượng đối tượng lưu vào biến toàn cục, trả ra chính đối tượng đó để luồng tiếp tục.
    """

    def __init__(
        self,
        label_name: str,
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]: # ! Confuse
        """
        Input: list_of_tracked_objects
        Output: int (số lượng đối tượng)
        """
        pass
    
    def update_db(self, target, input):
        return super().update_db(target, input)
    
    def update_cache(self, target, input):
        return super().update_cache(target, input)
    
    
class CheckAltitude(Core_Module):
    """
    Kiểm tra độ cao của một đối tượng so với mặt đất hoặc bề mặt tham chiếu.
    Nhận vào list tracked object và trả ra list tracked object nếu thỏa điều kiện.
    """

    def __init__(
        self,
        ref_surface: ZoneArea,
        condition: AltitudeCondition,
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        """
        Input: TrackedObject
        Output: TrackedObject (nếu điều kiện đúng)
        """
        pass
    
class CheckGroup(Core_Module):
    """
    Sử dụng thuật toán nhóm (clustering) để xác định các nhóm người dựa trên khoảng cách giữa họ.
    Module này sẽ nhận vào 1 list tracked object và trả ra 1 list tracked object
    List object mới này sẽ là tracked object của
    """

    def __init__(
        self,
        min_group_size: int,
        max_group_size: int,
        max_distance: float,
        label_in_group: Optional[List[str]] = None,
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]: # ! Confuse
        """
        Input: ListTrackedObject
        Output: ListTrackedObject
        """
        pass
    
    
class Counter(Updating_Module):
    """
    Tăng hoặc giảm bộ đếm dựa trên số tracked object mới xuất hiện
    Nhận vào tracked object và trả ra tracked object để luồng tiếp tục.
    """
    def __init__(
        self,
        varable_name: str,
        action: CounterAction,
    ) -> None:
        pass
    
    def update_cache(self, target, input):
        return super().update_cache(target, input)
    
    def update_db(self, target, input):
        return super().update_db(target, input)
    
# class ResetCounter(Updating_Module):
#     def __init__(
#         self,
#         varable_name: str,
#     ) -> None:
#         pass
    
#     def update_cache(self, target, input):
#         return super().update_cache(target, input)
    
#     def update_db(self, target, input):
#         return super().update_db(target, input)
    
class CheckCoMovement(Core_Module):
    """
    Kiểm tra xem hai đối tượng có di chuyển cùng nhau không dựa trên vận tốc
    và hướng di chuyển của chúng.
    Nhận vào cặp tracked object và trả ra cặp tracked object nếu thỏa điều kiện.
    Tính toán giữa các label trong list label ví dụ ['person', 'vehicle']
    Thông tin sẽ đựa lưu vào trong attribute của trackedobject1.predictions[n].attribute.comovement = {trackedobject2.track_id: (velocity_diff, direction_diff),
    """
    def __init__(
        self,
        time_window: int,
        velocity_tolerance: int,
        direction_tolerance: int,
        label: Optional[List[str]] = None,
    ) -> None:
        pass
    
    def process(
        self,
        pair: List[TrackedObject]
    ) -> List[TrackedObject]:
        pass
    
class CheckAbruptEvent(Core_Module):
    """
    Phát hiện sự thay đổi độ cao đột ngột của một đối tượng trong khoảng thời gian ngắn.
    """

    def __init__(
        self,
        vertical_displacement_threshold: float,  # mét
        time_window: float,  # giây
    ) -> None:
        pass

    def process(self, input: List[TrackedObject]) -> List[TrackedObject]:
        """
        Input: Tracked Object
        Output: Tracked Object (nếu điều kiện đúng)
        Logic:
          - Phân tích history của object.
          - Lấy điểm đáy bbox (bbox.y + bbox.height).
          - Nếu chênh lệch vị trí dọc trong Time Window > threshold → True.
        """
        pass
    
class CheckImpact(Core_Module):
    """
    Phát hiện va chạm giữa các đối tượng.
    Label = ['person', 'vehicle', 'MHE']
    Dựa vào khoảng cách và vận tốc của các đối tượng để xác định va chạm.
    Nhận vào list tracked object và trả ra list tracked object nếu thỏa điều kiện.    
    """
    def __init__(
        self,
        impact_sensitivity: ImpactSensitivity
    ) -> None:
        pass
    
    def process(
        self, 
        input: List[TrackedObject],
        label: Optional[List[str]] = None,
    ) -> List[TrackedObject]:
        pass
    
# class CheckReIdentification(Core_Module): # ! Confuse
#     """
#     So sánh đối tượng mới với các đối tượng đã biến mất gần đây để xác định xem có phải cùng một người không.
#     """

#     def __init__(
#         self, 
#         match_confidence_threshold: float, 
#         memory_duration: float = 5.0
#     ) -> None:
#         pass

#     def process(self, input: List[TrackedObject]) -> List[TrackedObject]: # ! Confuse
#         pass
    
class Evnet_Sender(Event_Module):
    """
    Module này nhận vào 1 list tracked object và trả ra 1 event.
    Trong list tracked object tiếng hành xử lý theo frame. Nếu trong 1 frame_id có nhiều tracked object, 
    ta gộp chúng lại thành 1 event.
    Sau khi trả ra event, module sẽ tự động gửi event đi.
    """
    def process(self, input: List[TrackedObject]) -> Event:
        return super().process(input)
    
class SingleViolationEventModule(Event_Module):
    def process(self, input: List[TrackedObject]) -> Event:
        return super().process(input)

class FrameSingleViolationEventModule(Event_Module):
    def process(self, input: Frame) -> Event:
        return super().process(input)

class SingleObjectEventModule(Event_Module):
    def process(self, input: List[TrackedObject]) -> Event:
        return super().process(input)

class FullFrameEventModule(Event_Module):
    def process(self, input: Frame) -> Event:
        return super().process(input)

get_input_from_model = InputFrame(
    frame_id=1,
    timestamp=1000,
    model_result=[
        {
            "tracked_id": 1,
            "bbox": [10, 20, 40, 20],
            "score": 0.8,
            "class_name": "staff",
            "keypoints": [],
            "segmentation": [],
            "color": None,
            "state": None,
            "action": None,
        },
        {
            "tracked_id": 2,
            "bbox": [10, 20, 40, 20],
            "score": 0.8,
            "class_name": "customer",
            "keypoints": [],
            "segmentation": [],
            "color": None,
            "state": None,
            "action": None,
        }
    ]
)

### Inference
tracked_object1 = TrackedObject(
    track_id=1,
    obj_name="staff",
    current_attribute= None,
    predictions=[
        Prediction(
            frame_id= 1,
            timestamp= 1000,
            detection= Detection(
                bbox = [10, 20, 40, 20],
                score = 0.8,
            ),
            segmentation= None,
            pose= None,
            attribute= Attribute(
                class_name = "staff",
                state = None,
                action = None,
                color = None
            )
        ),
        Prediction(
            frame_id= 2,
            timestamp= 1001,
            detection= Detection(
                bbox = [12, 22, 40, 20],
                score = 0.6,
            ),
            segmentation= None,
            pose= None,
            attribute= Attribute(
                class_name = "staff",
                state = None,
                action = None,
                color = None
            )
        )
    ]
)

tracked_object2 = TrackedObject(
    track_id=2,
    obj_name="customer",
    current_attribute = Attribute(
        state=None,
        action=None,
        color=None
        ),
    predictions=[
        Prediction(
            frame_id= 1,
            timestamp= 1000,
            detection= Detection(
                bbox = [10, 20, 40, 20],
                score = 0.8,
            ),
            segmentation= None,
            pose= None,
            attribute= Attribute(
                class_name = "customer",
                state = None,
                action = None,
                color = None
            )
        ),
        Prediction(
            frame_id= 2,
            timestamp= 1001,
            detection= Detection(
                bbox = [12, 22, 40, 20],
                score = 0.6,
            ),
            segmentation= None,
            pose= None,
            attribute= Attribute(
                class_name = "staff",
                state = None,
                action = None,
                color = None
            )
        )
    ]
)
frame = Frame(
    frame_id=1,
    timestamp=1000,
    list_tracked_object=[
        tracked_object1,
        tracked_object2
    ]
)
# Inference 1 NOBARTENDER

node1 = Trigger_Kafka_Message()
node2 = Filter_Objects(
    value=['customer'],
    condition=Comparison.EQUAL
)

node3 = Filter_Objects(
    value=['staff'],
    condition=Comparison.EQUAL
)

node4 = CheckPose(
    pose_condition=PoseCondition.RAISE_HAND
)

node5 = CheckGroup(
    min_group_size=2,
    max_group_size=2,
    max_distance=1.5,
    label_in_group=['customer', "staff"]
    
)

node6 = IF(
    code_block=CodeBlock()
)

node7 = Timer(
    timer_name="timer_no_bartender",
    condition=TimeCondition.LONGER_THAN,
    duration=300.0
)

node8 = FrameSingleViolationEventModule()


node1.add_next(node2)

node2.add_next(node4)

node3.add_next(node5)

node4.add_next(node5)

# Chờ cho đến khi node 5 nhận được đủ 2 luồng input (2 node trước đó thực thi xong)
node5.add_next(node6)

# Có group rồi thì node 7, chưa có group thì node 8
node6.add_next([node7, node8]) # True → node7, False → node8

node7.add_next(node8)


# Inference 2 30MINCHECK

# Workflow 1
n1 = Trigger_Kafka_Message()
n2 = Filter_Objects(
    value=["manager"],
    condition=Comparison.EQUAL
)
n3 = CheckZone(
    zone=[
        ZoneArea(
        zone_id="kho1",
        polygon= None # Full screen
        ),
        ZoneArea(
        zone_id="kho2",
        polygon= None # Full screen
        ),
        ZoneArea(
        zone_id="kho3",
        polygon= None # Full screen
        ),
    ],
    condition=ZoneCondition.IS_INSIDE
)

n4 = InitUpdateVariable(
    variable_name="kiem_tra_kho",
    action= VariableAction.INCREMENT,
    value=None
)


n1.add_next(n2)
n2.add_next(n3)
n3.add_next(n4)


# Inference 3 Collision

n1 = Trigger_Kafka_Message()

n2 = Filter_Objects(
    value=['person', "MHE"],
    condition=Comparison.EQUAL
)

n3 = CheckDistance(
    distance=2.0,
    condition=Comparison.LESS_THAN
)

n4 = FrameSingleViolationEventModule()

n5 = Filter_Objects(
    value="MHE",
    condition=Comparison.EQUAL
)

n6 = CheckDistance(
    distance=1.0,
    condition=Comparison.LESS_THAN
)

n7 = FrameSingleViolationEventModule()

n1.add_next([n2, n5])

n2.add_next(n3)
n3.add_next(n4)

n5.add_next(n6)
n6.add_next(n7)

# Inference 4 Drive Exit
n1 = Trigger_Kafka_Message()

n2 = Filter_Objects(
    value=['vehicle', 'driver'],
)

n3 = CheckZone(
    zone=[
        ZoneArea(
        zone_id="exit_area",
        polygon= [100, 100, 100, 100] # Box của các driver
        ),
    ]
)

n4 = CheckDistance(
    distance=5.0,
    condition=Comparison.LESS_THAN
)

n5 = Logic_Code(
    code_block=CodeBlock()
)

n6 = FrameSingleViolationEventModule()

n1.add_next(n2)
n2.add_next(n3)
n3.add_next(n4)
n4.add_next(n5)
n5.add_next(n6)


