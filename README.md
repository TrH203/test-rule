## **Những việc đã hoàn thành**

### **Giai đoạn khung xử lý (Pipeline foundation)**

* [x] **Tạo pseudo code** mô tả toàn bộ luồng xử lý dữ liệu qua các node/module
: `pseudo_module_1.py`, `pseudo_module_2.py`
* [x] **Viết code chuẩn (all_modules.py)** để định nghĩa rõ:

  * Kiểu dữ liệu (`Detection`, `Pose`, `Segmentation`, `Attribute`, `Prediction`, `TrackedObject`, `Event`, `Context`)
  * Cấu trúc pipeline và cơ chế clone, cập nhật dữ liệu theo frame.


### **Dữ liệu và đầu vào và ra**

* [x] **Tạo data mẫu từ model (InputFrame)**
  -> mô phỏng kết quả đầu ra từ mô hình AI: `simulated_model_input.json`  (bbox, class, ppe, score, …)
* [x] **Tạo data mẫu từ rule (Event)**
  -> mô phỏng kết quả kiểm tra điều kiện hoặc vi phạm được phát hiện : `simulated_contexts.json`


### **Các module xử lý**

* [x] **Tạo xử lý sau của các module**
  -> định nghĩa cách module cập nhật `Prediction.attribute` và `current_attribute` sau khi tính toán.
* [x] **Tạo simulation module Filter Object**
  -> lọc object đủ điều kiện đi qua pipeline (ví dụ: lọc class hoặc score thấp).
* [x] **Tạo simulation module Check Zone**
  -> kiểm tra object có nằm trong khu vực xác định (ZoneArea) hay không.
* [x] **Tạo simulation module Timer**
 -> theo dõi thời gian tồn tại, thời lượng vi phạm, hoặc duy trì trạng thái liên tục qua nhiều frame.


## **Trạng thái hiện tại**

Hiện pipeline của bạn đã hoàn chỉnh **từ khâu đầu vào -> xử lý -> lưu kết quả -> mô phỏng rule/event**, bao gồm:

* Dữ liệu khởi tạo (InputFrame -> Context)
* Cơ chế enrich dữ liệu (Attribute)
* Xử lý trạng thái liên tục (Timer)
* Kiểm tra điều kiện vùng (Zone)
* Bộ lọc và rule kiểm tra vi phạm


## 1. Tầng đầu vào (InputFrame)

`InputFrame` là dữ liệu đầu tiên được nhận từ mô hình AI (YOLO, Pose, Segmentation, v.v.) tương ứng **một frame trong video**.

* Mỗi phần tử trong `model_result` đại diện cho **một đối tượng được phát hiện** trong frame đó.
* Các trường có thể **đầy đủ hoặc thiếu** tùy vào mô hình:

  * `tracked_id`, `bbox`, `class_name`, `keypoints`, `segmentation`, `score`, `ppe_status`, `state`, `action`, `color`, ...
* Đây là dữ liệu **thô từ model**, chưa tổ chức theo chuỗi thời gian hay logic.


## 2. Chuyển đổi sang Context

Sau khi nhận `InputFrame`, dữ liệu được **chuyển đổi thành Context**, là cấu trúc trung tâm lưu trữ toàn bộ trạng thái pipeline

Code: `convert_input_model_to_context.py`

```python
class Context:
    frame_id: int
    timestamp: int
    tracked_objects: List[TrackedObject]
    global_vars: Dict[str, Any]
    events: List[Event]
```

### Vai trò:

* Là **container chính** mang dữ liệu qua từng node xử lý trong pipeline.
* Dữ liệu trong `Context` có thể được **copy độc lập (clone)** để xử lý song song, tránh xung đột.



## 3. Cấu trúc bên trong Context

### `TrackedObject`

Đại diện cho **một thực thể theo dõi xuyên suốt các frame** (ví dụ: một người hoặc xe).

```python
TrackedObject(
    track_id=...,
    obj_name="staff",
    current_attribute=Attribute(...),
    predictions=[Prediction(frame1), Prediction(frame2), ...]
)
```

* `track_id` giúp nhận diện cùng đối tượng qua nhiều frame.
* `predictions`: lưu toàn bộ lịch sử xuất hiện theo thời gian.


## 4. Dữ liệu dự đoán (Prediction)

Mỗi `Prediction` tương ứng **một frame** của đối tượng đó:

```python
Prediction(
    frame_id=1,
    timestamp=1000,
    detection=Detection(...),
    pose=Pose(...),
    segmentation=Segmentation(...),
    attribute=Attribute(...)
)
```

Nó chứa kết quả nhận diện, pose, segmentation, và các thuộc tính được tính từ mô hình hoặc các node khác.


## 5. Thuộc tính mở rộng (Attribute)

`Attribute` là nơi lưu toàn bộ **thông tin động** hoặc **tính toán thêm** của đối tượng:

* Thông tin PPE (`helmet`, `vest`, `safety_shoes`)
* Trạng thái (`state`, `action`)
* Tốc độ, hướng di chuyển, độ cao
* Tương tác nhóm (`group_id`, `nearby_objects`, `comovement`)
* Bộ đếm (`timers`): dùng cho rule detection hoặc event trigger

---

## 6. Tích lũy và lưu trữ theo thời gian

* Khi **frame mới** đến -> tìm `TrackedObject` có cùng `track_id` -> **append thêm `Prediction` mới** vào danh sách `predictions`.
* Nhờ đó, mỗi `TrackedObject` trở thành **lịch sử toàn bộ hành vi** của một đối tượng theo thời gian.
* Dữ liệu lưu tạm trong `Context`, có thể ghi ra file/db hoặc dùng cho rule engine.

---

## 7. Tóm tắt quan hệ dữ liệu

```
InputFrame (raw model output)
     ↓ convert
Context
 ├── frame_id / timestamp
 ├── tracked_objects: List[TrackedObject]
 │      ├── track_id
 │      ├── current_attribute
 │      └── predictions: List[Prediction]
 │             ├── detection / pose / segmentation
 │             └── attribute
 ├── events (phát sinh bởi rule)
 └── global_vars (thông tin chung toàn frame)
```

---

**Tóm lại:**

* `InputFrame`: đầu vào thô theo frame.
* `Context`: bộ khung chứa dữ liệu pipeline.
* `TrackedObject`: theo dõi đối tượng theo thời gian.
* `Prediction`: dữ liệu chi tiết theo từng frame.
* `Attribute`: đặc tính mở rộng động.



# Hướng thiết kế pipeline theo dạng Stateful Context

**Hướng thiết kế pipeline theo dạng Stateful Context**, nơi **mỗi module không chỉ lọc mà còn làm giàu dữ liệu theo thời gian**.


## 1. Dữ liệu đầu vào và khởi tạo

* Frame đến -> tạo `InputFrame` -> convert thành `Context`.
* Mỗi object trong frame sinh ra 1 `TrackedObject` (nếu chưa có) hoặc cập nhật đối tượng đã tồn tại (nếu cùng `track_id`).
* `Prediction` mới được tạo và append vào `TrackedObject.predictions`.

---

## 2. Cách các module xử lý dữ liệu

Các **module** không chỉ “lọc” mà còn **tính toán, bổ sung và ghi nhận thông tin** tại từng frame:

| Loại tính toán                     | Nhiệm vụ chính                                                         | Kết quả lưu ở đâu                                 |
| -------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------- |
| **Identity**          | Liên kết frame hiện tại với tracked_id cũ                              | TrackedObject                                     |
| **Geometry / Speed / Direction** | Tính vận tốc, hướng dựa trên vị trí liên tiếp giữa các frame           | `Prediction.attribute.speed`, `direction`         |
| **Group / Nearby**               | Tính khoảng cách giữa các object trong cùng frame                      | `Prediction.attribute.nearby_objects`, `group_id` |
| **Rule / Event**                 | Phân tích hành vi hoặc trạng thái, trigger sự kiện                     | `Context.events`                                  |
| **Aggregation**                  | Tính trung bình hoặc cập nhật giá trị tổng hợp cho `current_attribute` | `TrackedObject.current_attribute`                 |

---

## 3. Lưu dữ liệu theo thời gian (Per-frame computation)

Mỗi frame sinh ra **một “bản ghi” trạng thái** cho từng object:

```
TrackedObject.predictions = [
    Prediction(frame1, attribute=...),
    Prediction(frame2, attribute=...),
    ...
]
```

-> `Prediction.attribute` chứa **toàn bộ kết quả tính toán của frame đó** (gốc + module).
-> `current_attribute` là **bản tóm tắt hoặc giá trị cuối cùng**, ví dụ:

* Lấy trung bình (`mean speed, mean direction`)
* Hoặc lấy giá trị gần nhất (`latest state, latest color`)

Điều này giúp bạn:

* Truy ngược lịch sử: biết tại frame nào object có PPE, hướng, nhóm ra sao.
* Cập nhật liên tục mà không mất dữ liệu cũ.

---

## 4. Khi đối tượng vi phạm liên tục

Ví dụ bạn có **Event Rule**: “Không đội mũ bảo hộ trong khu vực cấm”.

* Ở **frame 1**, nếu PPE không đạt -> Event được tạo.
* Ở **frame 2**, nếu vẫn không đạt -> module vẫn ghi vi phạm mới, nhưng có thể:

  * **Gộp** với vi phạm cũ (nếu cùng `track_id` và gần timestamp), hoặc
  * **Tăng thời lượng/tần suất vi phạm** trong `attribute.timers["violation"] += Δt`.

Như vậy: **Event node không cần tạo sự kiện mới liên tục**, mà có thể nhìn vào lịch sử `Prediction.attribute` để quyết định khi nào là “vẫn vi phạm”, “liên tục”, hay “đã khắc phục”.

---

## 5. Cơ chế lưu và cập nhật

Tóm lại, dữ liệu được lưu theo hai tầng:

| Tầng                                 | Mục đích                                 | Dạng dữ liệu                              |
| ------------------------------------ | ---------------------------------------- | ----------------------------------------- |
| **Per-frame (Prediction.attribute)** | Ảnh chụp trạng thái tại thời điểm cụ thể | Chi tiết, phục vụ phân tích               |
| **Cộng dồn (current_attribute)**     | Tình trạng hiện tại hoặc trung bình      | Dễ đọc, dùng cho hiển thị / trigger logic |
| **Context.events**                   | Lưu các vi phạm hoặc sự kiện phát sinh   | Cấu trúc riêng cho rule                   |

---

## 6. Tổng thể pipeline hoạt động

```
InputFrame -> Context(frame_id)
    ↓
[Module 1] Detection -> enrich bbox/class
    ↓
[Module 2] Tracking -> assign track_id
    ↓
[Module 3] Motion -> speed/direction
    ↓
[Module 4] Group/Nearby -> distance/group
    ↓
[Module 5] Rule/Event -> check violation
    ↓
Cập nhật Context:
- tracked_objects.predictions[*].attribute
- tracked_objects.current_attribute
- context.events[*]
```

---

**Tóm lại**,

* Pipeline có **trạng thái tích lũy (stateful)**.
* Dữ liệu **vừa được làm giàu theo thời gian (temporal enrichment)**, vừa **được snapshot mỗi frame (Prediction.attribute)**.
* `current_attribute` giữ **tình trạng tức thời**, `predictions` giữ **lịch sử đầy đủ**, còn `events` là **hệ quả logic cuối**.

