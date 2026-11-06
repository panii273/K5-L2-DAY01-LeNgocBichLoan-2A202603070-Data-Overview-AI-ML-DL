# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026**

**Runtime Colab: GPU T4(Device: cuda)** 

**Python / PyTorch / Ultralytics:Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:0.35(giữ nguyên, không đổi)** 

**Thay đổi so với notebook nguồn: không đổi mã nguồn, không đổi checkpoint hay ngưỡng** 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
class_id=468 ; class_name=cab ; rank=1 ; score=0.510915 ; taxonomy_name=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh vì trong record chỉ có class_id, class_name, rank hay score, không có box hay tọa độ nào chỉ vào một chỗ trong ảnh. Còn ảnh traffic có rất nhiều ảnh xe bus thực tế, ô tô và người nhưng model chỉ đưa ra một lớp cho cả ảnh, không chỉ từng vật.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách đi kèm với taxonomy/checkpoint, model không tự nghĩ ra. Model yolo11n-cls.pt dùng taxonomy ImageNet-1K (đúng như trường taxonomy_name). Danh sách này do con người chọn từ trước, model chỉ học cách xếp hạng trong danh sách đó.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Nếu chỉ ghi tên lớp thì dễ hiểu nhầm. class_id là số máy dùng, class_name là tên người đọc, còn taxonomy_name cho biết số và tên đó thuộc bảng nào. Cùng một con số có thể có nghĩa khác nhau ở bảng khác (ví dụ ImageNet-1K và COCO-80 đánh số khác nhau). Ghi đủ ba thứ thì sau này ai đọc cũng biết đúng nghĩa.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần nói rõ cách chọn một lớp khi ảnh có nhiều vật: chọn vật to nhất, vật nổi bật nhất, hay chọn theo chủ đề chung của cảnh. Guideline cũng cần nói khi hai lựa chọn ngang nhau thì làm gì, và khi nào thì hỏi Lab Coach. Ví dụ ảnh traffic có xe buýt, ô tô và người đi bộ. Nếu guideline không nói rõ thì mỗi người gán sẽ chọn một lớp khác nhau.
6. Vì sao không được lấy `score` của model làm ground truth?
score chỉ là mức "tin" của chính model về dự đoán của nó, chưa có người nào kiểm tra. Điểm cao không có nghĩa là đúng. Ví dụ ảnh này có rất nhiều xe buýt, nhưng model xếp cab đứng đầu với score 0.51, còn minibus chỉ 0.16. Ground truth phải do người gán theo guideline rồi có reviewer kiểm tra lại, không lấy từ score của model.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
Chọn record person ): class_name=person ; score=0.912625 ; bbox_xyxy=[385.33, 69.24, 498.92, 348.92] ; bbox_width=113.58 ; bbox_height=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
bbox_xyxy gồm 4 số theo thứ tự [x_min, y_min, x_max, y_max], tính bằng pixel, và gốc (0, 0) nằm ở góc trên-trái của ảnh. Với hộp này:
Góc trên-trái ở điểm (385.33, 69.24): cách mép trái ảnh 385.33 pixel và cách mép trên 69.24 pixel.
Góc dưới-phải ở điểm (498.92, 348.92).
Hộp rộng 113.58 pixel (498.92 trừ 385.33) và cao 279.68 pixel (348.92 trừ 69.24). Con số trong JSON là 113.58 chứ không phải 113.59 vì máy làm tròn từ số gốc chưa làm tròn.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
threshold=0.20: 17 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']
threshold=0.60: 6 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Hạ ngưỡng thì có nhiều hộp hơn, nhưng nhiều hộp trong đó model không chắc nên dễ sai (sai lớp, hoặc không phải vật nào cả). Vì vậy reviewer phải kiểm tra thêm từng hộp mới: hộp có bao đúng vật không, lớp có đúng không, có hộp nào trùng nhau không, rồi giữ, sửa hoặc xóa. Càng nhiều hộp thì reviewer càng mất nhiều việc.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
vẽ một box chữ nhật vừa khít quanh phần nhìn thấy của vật. Các cạnh box chạm sát mép ngoài của vật, không chừa khoảng trống thừa và không cắt mất một phần vật. Mỗi vật chỉ vẽ một box. Quy tắc này dùng được cho mọi ảnh, không riêng ảnh bếp.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Ở mép trái ảnh kitchen có một box person (score 0.610931, bbox_xyxy=[0.12, 263.18, 61.55, 310.78]) mà chỉ thấy một phần cánh tay vì bị mép ảnh cắt mất. Số x_min = 0.12 gần bằng 0 nghĩa là box chạm sát mép trái của ảnh. Trường hợp này tôi sẽ đưa lên hỏi Lab Coach (escalation) nếu guideline chưa nói gì về vật chỉ lộ một phần. Lý do là có hai cách làm đều hợp lý: vẫn gán nhãn cho phần nhìn thấy, hoặc bỏ qua vì quá nhỏ. Nếu mỗi người tự chọn một cách thì nhãn sẽ không đồng nhất. Sau khi Lab Coach trả lời thì mới ghi thành quy tắc trong guideline.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
record person: instance_id=kitchen-001 ; class_name=person ; score=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 5 điểm đầu của polygon_xy=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding box chỉ cung cấp hình chữ nhật bao quanh, bao gồm cả các pixel nền nằm bên trong hộp. Polygon gồm nhiều điểm nối thành đường biên gần với hình dạng đối tượng, nhờ đó thể hiện vùng chiếm chỗ của đối tượng chính xác hơn, đặc biệt khi vật có dạng không đều. Tuy vậy, polygon do model dự đoán vẫn có thể lệch biên và cần kiểm tra.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
Đây là mã định danh một đối tượng cụ thể trong tập prediction đã xuất. Hai vật cùng lớp (ví dụ hai bowl là kitchen-002 và kitchen-003) vẫn phải có instance_id và mask riêng. instance_id khác class_id (mã loại vật) và không phải tracking_id (mã theo dõi cùng một vật qua nhiều khung hình/video).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
vẽ đa giác bám sát đường viền thật của vật, gồm toàn bộ phần nhìn thấy, không lấn sang nền hay sang vật khác. Nếu vật bị che thì chỉ vẽ phần nhìn thấy, không đoán phần bị che. Mỗi vật vẽ một đa giác riêng.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Ở góc dưới bên trái ảnh kitchen, có nhiều vật thể bowl nằm sát và chồng lấp lên nhau, khiến ranh giới giữa các vật thể khó xác định.
    Trong trường hợp này, tôi sẽ áp dụng Guideline nếu vẫn có thể nhận diện phần nhìn thấy của từng vật thể. Theo đó, mỗi vật được vẽ bằng một polygon riêng và chỉ bao phủ phần nhìn thấy.
   Tuy nhiên, nếu ranh giới giữa các vật thể quá mờ và không thể xác định chính xác, tôi sẽ thực hiện Escalation với Lab Coach để được hướng dẫn. Điều này giúp đảm bảo tính nhất quán và độ chính xác của dữ liệu annotation.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một nhãn lớp ở cấp toàn ảnh nếu guideline quy định single-label; có mã/tên lớp và taxonomy.| Traffic chứa nhiều xe, nhưng model xếp cab hạng 1 với score 0.510915; nhãn toàn ảnh dễ mơ hồ khi có nhiều chủ thể. | Quan sát toàn ảnh, đối chiếu mục tiêu và tiêu chí chủ thể chính, gán nhãn theo taxonomy; gắn cờ trường hợp khó. |Kiểm tra nhãn có mô tả đúng toàn ảnh không, có nhất quán guideline và taxonomy không; không dựa độc nhất vào score.  |
| Phát hiện vật thể |Mỗi đối tượng một class và bounding box xyxy theo pixel.  | Trên PNG kitchen, một người chỉ lộ một phần ở mép trái (person, score 0.610931; bbox [0.12, 263.18, 61.55, 310.78]); nhiều hộp bowl nhỏ gần khu vực treo dụng cụ cần kiểm tra nhầm lớp. |Vẽ hộp chặt, gán lớp, tìm vật bị bỏ sót, ghi chú trường hợp bị che/cắt mép theo guideline.  | Kiểm tra đúng lớp, đủ vật, hộp không quá rộng/hẹp, không trùng và các ca che khuất được xử lý thống nhất. |
| Instance segmentation |Mỗi đối tượng có class, instance_id và polygon/mask riêng.  |PNG cho thấy dining table (kitchen-005, 598 điểm polygon) có mask trải rộng cả mặt bàn và phần cấu trúc bên dưới, cần rà phần giáp đồ vật khác. Mask oven 0.49 chồng lên person 0.43 ở góc trái dưới, và mask potted plant 0.63 phủ lên bó rau thơm treo (nhiều khả năng nhầm lớp), đều là ca cần QC.  |Vẽ biên theo phần nhìn thấy, tách từng instance, không tô vùng của vật khác; chuyển ca khó khi cần.  |Kiểm tra mask sát biên, không nhập hai vật làm một, không bỏ vùng đối tượng; kiểm tra tính nhất quán giữa các instance.  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
chỉ dùng ba ảnh COCO công khai mà notebook có sẵn, không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub, và không ghi họ tên hay MSSV vào báo cáo hoặc output (chỉ có trong tên repository).
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, mình sẽ dừng lại, không dùng tiếp và báo cho Lab Coach. Nếu lỡ để dữ liệu cá nhân trong báo cáo hay output, mình sẽ xóa đi trước khi nộp link, và nếu đã push lên GitHub thì báo thêm cho giảng viên.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
