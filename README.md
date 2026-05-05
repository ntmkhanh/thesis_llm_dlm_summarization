# thesis_llm_dlm_summarization

README này mô tả **đúng flow đề cương luận văn**:
1. Fine-tune LLM baseline trên CNN-only.
2. Phương pháp 1 (LLM -> DLM):
   - Single-Draft (Hình 3)
   - Multi-Draft (i) refine từng draft rồi chọn (Hình 4)
   - Multi-Draft (ii) latent aggregation rồi reverse diffusion (Hình 5)
     - bản `mean`
     - bản `learned latent fusion` (đề xuất mới)
3. Phương pháp 2 (DLM -> LLM decoder) theo 3 bước (Hình 6).
4. Đánh giá ROUGE/BERTScore.

## A. Dataset và split (đúng đề cương)

- Dataset: `cnn_dailymail` version `3.0.0`
- Chỉ dùng mẫu CNN (`article` bắt đầu bằng `(CNN)`)
- Split cố định:
1. `train`
2. `validation`
3. `test`

## B. Setup môi trường

```bash
git clone https://github.com/ntmkhanh/thesis_llm_dlm_summarization.git
cd thesis_llm_dlm_summarization
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Kiểm tra GPU:
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## C. Bước 1 - Train baseline LLM (Summary Generator)

Theo đề cương: fine-tune LLM trên CNN/DailyMail để tạo model sinh tóm tắt cơ sở.

```bash
python3 src/pipeline/train_llm_sft.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train-split train \
  --val-split validation \
  --output-dir outputs/models/llm_sft \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8
```

## D. Bước 2 - Phương pháp 1: LLM -> DLM

---
### D1) Train DLM refiner (paper-aligned, text-level diffusion)

Dùng cho Single-Draft + Multi-Draft (i)/(ii) bản practical.

```bash
python3 src/pipeline/train_dlm_refiner.py \
  --model google/flan-t5-base \
  --paper-mode seqdiffuseq \
  --diffusion-steps 8 \
  --train-split train \
  --val-split validation \
  --output-dir outputs/models/dlm_refiner \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 4
```

---
### D2) Single-Draft (Hình 3)

```bash
python3 src/pipeline/infer_method1_llm_dlm.py \
  --llm-model-dir outputs/models/llm_sft \
  --dlm-model-dir outputs/models/dlm_refiner \
  --paper-mode seqdiffuseq \
  --diffusion-steps 8 \
  --draft-mode single \
  --split test \
  --output outputs/drafts/method1_single.csv
```

---
### D3) Multi-Draft (i) refine từng draft rồi chọn tốt nhất (Hình 4)

```bash
python3 src/pipeline/infer_method1_llm_dlm.py \
  --llm-model-dir outputs/models/llm_sft \
  --dlm-model-dir outputs/models/dlm_refiner \
  --paper-mode seqdiffuseq \
  --diffusion-steps 8 \
  --draft-mode multi_refine_each \
  --num-candidates 3 \
  --split test \
  --output outputs/drafts/method1_multi_refine_each.csv
```

---
### D4) Multi-Draft (ii) latent aggregation trước reverse diffusion (Hình 5)

#### D4.1 Bản baseline: `mean latent fusion`
```bash
python3 src/pipeline/infer_method1_llm_dlm.py \
  --llm-model-dir outputs/models/llm_sft \
  --dlm-model-dir outputs/models/dlm_refiner \
  --paper-mode seqdiffuseq \
  --diffusion-steps 8 \
  --draft-mode multi_aggregate_latent \
  --latent-fusion mean \
  --num-candidates 3 \
  --split test \
  --output outputs/drafts/method1_multi_aggregate_latent_mean.csv
```

#### D4.2 Bản đề xuất mới: `learned latent fusion`
Train gating:
```bash
python3 src/pipeline/train_latent_fusion_gating.py \
  --llm-model-dir outputs/models/llm_sft \
  --dlm-model-dir outputs/models/dlm_refiner \
  --num-candidates 3 \
  --max-samples 5000 \
  --output outputs/models/latent_fusion/gating.pt
```

Infer learned:
```bash
python3 src/pipeline/infer_method1_llm_dlm.py \
  --llm-model-dir outputs/models/llm_sft \
  --dlm-model-dir outputs/models/dlm_refiner \
  --paper-mode seqdiffuseq \
  --diffusion-steps 8 \
  --draft-mode multi_aggregate_latent \
  --latent-fusion learned \
  --latent-fusion-model outputs/models/latent_fusion/gating.pt \
  --num-candidates 3 \
  --split test \
  --output outputs/drafts/method1_multi_aggregate_latent_learned.csv
```

## E. Bước 3 - Phương pháp 1 bản latent diffusion thuần (mở rộng nghiên cứu)

Phần này là bản tách module rõ ràng theo yêu cầu mở rộng:
- `src/dlm/core_diffusion.py` (schedule + q/p sampler)
- `src/dlm/latent_denoiser.py` (timestep-conditioned denoiser)
- `src/pipeline/train_dlm_latent.py`
- `src/pipeline/infer_method1_latent.py`

### E1) Train latent diffusion thuần
```bash
python3 src/pipeline/train_dlm_latent.py \
  --model google/flan-t5-base \
  --train-split train \
  --val-split validation \
  --timesteps 100 \
  --self-condition \
  --output-dir outputs/models/dlm_latent
```

### E2) Inference so sánh `mean` vs `learned` trên latent diffusion thuần

Mean:
```bash
python3 src/pipeline/infer_method1_latent.py \
  --llm-model-dir outputs/models/llm_sft \
  --latent-model-dir outputs/models/dlm_latent \
  --fusion mean \
  --num-candidates 3 \
  --split test \
  --output outputs/drafts/method1_latent_mean.csv
```

Learned:
```bash
python3 src/pipeline/infer_method1_latent.py \
  --llm-model-dir outputs/models/llm_sft \
  --latent-model-dir outputs/models/dlm_latent \
  --fusion learned \
  --gating-model outputs/models/latent_fusion/gating.pt \
  --num-candidates 3 \
  --split test \
  --output outputs/drafts/method1_latent_learned.csv
```

## F. Bước 4 - Phương pháp 2: DLM -> LLM Decoder (Hình 6)

### F1) Train Planner (article -> plan)
```bash
python3 src/pipeline/train_method2_planner.py \
  --model google/flan-t5-base \
  --train-split train \
  --val-split validation \
  --output-dir outputs/models/method2_planner \
  --epochs 1
```

### F2) Train Decoder (article+plan -> summary)
```bash
python3 src/pipeline/train_method2_decoder.py \
  --model google/flan-t5-base \
  --train-split train \
  --val-split validation \
  --output-dir outputs/models/method2_decoder \
  --epochs 1
```

### F3) Infer Method 2
```bash
python3 src/pipeline/infer_method2_dlm_llm.py \
  --planner-model-dir outputs/models/method2_planner \
  --decoder-model-dir outputs/models/method2_decoder \
  --split test \
  --output outputs/drafts/method2_dlm_llm.csv
```

## G. Bước 5 - Đánh giá (ROUGE + BERTScore)

```bash
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_single.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_multi_refine_each.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_multi_aggregate_latent_mean.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_multi_aggregate_latent_learned.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_latent_mean.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_latent_learned.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method2_dlm_llm.csv
```

## H. So sánh baseline cũ vs pipeline mới

Các script cũ vẫn giữ để baseline kỹ thuật:
- `src/llm/generate_baseline.py`
- `src/llm/generate_method1.py`
- `src/llm/generate_method2.py`

Bạn có thể chạy lại các script cũ để đối chiếu với pipeline mới ở `src/pipeline/*`.

## I. Mẹo chạy nhanh debug

- Thêm `--max-train-samples`, `--max-val-samples`, `--max-samples` để thử nhanh.
- Sau khi ổn mới chạy full `train/validation/test`.
- Nên lưu mỗi run vào file output khác nhau để không ghi đè.
