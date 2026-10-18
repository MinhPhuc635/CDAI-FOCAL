# FOCAL với Backbone DINOv3

Bản fork của [tomoyoshki/focal](https://github.com/tomoyoshki/focal) (NeurIPS 2023, [arXiv:2310.20071](https://arxiv.org/abs/2310.20071)).

**Điểm thay đổi:** encoder (backbone) được thay bằng **DINOv3**. Các hàm loss của FOCAL, pipeline dữ liệu và quy trình huấn luyện giữ nguyên. Về phương pháp của bài báo, các dataset, cách tiền xử lý và hai backbone gốc (DeepSense, Swin-Transformer), xem [repo gốc](https://github.com/tomoyoshki/focal).

## Tóm tắt các thay đổi

| File | Thay đổi |
|---|---|
| `src/models/DINOv3.py` | **Mới.** Backbone DINOv3, chọn bằng `-model=DINOv3`. |
| `src/train_utils/model_selection.py` | Import `DINOv3` và thêm nhánh `elif args.model == "DINOv3"` trong hàm `init_backbone_model`. |
| `src/data/MOD.yaml` | Thêm khối cấu hình `DINOv3:` và hai mục theo tên model trong `FOCAL`; hạ learning rate của pretrain. |

### Cách backbone hoạt động (`src/models/DINOv3.py`)

Với mỗi modality, đầu vào FFT có dạng `[b, c, i, s]` đi qua các bước:

1. biến đổi signed-log trên phần thực/ảo (kèm thêm magnitude),
2. một lớp tích chập 1x1 có thể học, đưa số kênh về 3 (giống ảnh RGB),
3. resize song tuyến tính (bilinear) về `img_size x img_size`,
4. encoder DINOv3 (mỗi modality một encoder riêng), lấy token CLS,
5. một lớp linear đưa về `loc_out_channels`.

Backbone có cùng giao diện với `SW_Transformer` (`forward(freq_x, class_head, proj_head)`), nên không cần sửa loss hay code huấn luyện của FOCAL.

### Thay đổi cấu hình (`src/data/MOD.yaml`)

- Thêm khối `DINOv3:` ở cấp cao nhất, gồm `pretrained_name`, `img_size` (112), `loc_out_channels` (256), `pretrained_head`, `optimizer`, `lr_scheduler`, `random_augmenters`, `fixed_augmenters`.
- Thêm `DINOv3: 0.07` dưới `FOCAL -> temperature` và `DINOv3: 0.005` dưới `FOCAL -> finetune_optimizer -> weight_decay`. Cả hai bảng được tra theo tên model; thiếu mục này sẽ báo lỗi `KeyError: 'DINOv3'`.
- Hạ `FOCAL -> pretrain_optimizer -> start_lr` từ `0.001` xuống `0.0001` để tránh làm hỏng trọng số DINOv3 đã được pretrain.

## Cài đặt

Cài các thư viện như repo gốc, sau đó cài thêm:

```bash
pip install -U "transformers>=4.56" huggingface_hub
```

Trên Google Colab, **không** chạy `pip install -r requirements.txt` (file này ghim phiên bản PyTorch/CUDA cũ và xung đột với môi trường Colab). Chỉ cài những gói cần thiết:

```bash
pip install -q "transformers>=4.56" tsai timm einops scikit-learn pyyaml tqdm huggingface_hub
```

### Trọng số DINOv3 (model có giới hạn truy cập)

1. Tạo tài khoản Hugging Face và đồng ý điều khoản sử dụng trên trang model, ví dụ `facebook/dinov3-vits16-pretrain-lvd1689m`.
2. Tạo access token loại *Read* (Settings -> Access Tokens).
3. Đăng nhập trước khi chạy code:

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")   # trên Colab: lưu token trong Secrets với tên HF_TOKEN
```

Nếu chưa được cấp quyền truy cập, có thể dùng `facebook/dinov2-small` (không bị giới hạn) thay thế bằng cách đổi `pretrained_name` trong `MOD.yaml`.

## Cách sử dụng (DINOv3)

Chạy mọi lệnh từ thư mục `src/`. Việc chuẩn bị dữ liệu và các file index giống như repo gốc.

```bash
# 1. Pretrain
python train.py -model=DINOv3 -dataset=MOD -learn_framework=FOCAL -batch_size=64

# 2. Fine-tune
python train.py -model=DINOv3 -dataset=MOD -learn_framework=FOCAL \
    -task=vehicle_classification -stage=finetune \
    -model_weight=../weights/MOD_DINOv3/exp0_contrastive_FOCAL

# 3. Test
python test.py -model=DINOv3 -dataset=MOD -stage=finetune \
    -task=vehicle_classification \
    -model_weight=../weights/MOD_DINOv3/exp0_contrastive_FOCAL
```

Trọng số được lưu trong `weights/MOD_DINOv3/`. Lần chạy đầu tiên tạo thư mục `exp0_contrastive_FOCAL`, các lần sau tạo `exp1_...`, `exp2_...`.

### Lưu ý khi dùng GPU 16 GB (ví dụ Colab T4)

- Batch size mặc định
