# thesis_llm_dlm_summarization

README này mô tả **flow đề cương luận văn**:
1. Fine-tune LLM baseline trên CNN-only.
2. Phương pháp 1 (LLM -> DLM):
   - Single-Draft (Hình 3)
   - Multi-Draft (i) refine từng draft rồi chọn (Hình 4)
   - Multi-Draft (ii) latent aggregation rồi reverse diffusion (Hình 5)
     - bản `mean`
     - bản `learned latent fusion` (đề xuất mới)
3. Phương pháp 2 (DLM -> LLM decoder) theo 3 bước (Hình 6).
4. Đánh giá ROUGE/BERTScore.

## A. Dataset và split

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
python3 src/pipeline/train_llm_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --tuning-mode qlora --train-split train --val-split validation --max-length 1024 --batch-size 1 --grad-accum 16 --fp16 --gradient-checkpointing --output-dir outputs/models/llm_sft_qlora --epochs 90
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

## E. Bước 3 - Phương pháp 1 bản latent diffusion thuần (mở rộng)

Phần này là bản tách module:
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

Các script cũ vẫn giữ:
- `src/llm/generate_baseline.py`
- `src/llm/generate_method1.py`
- `src/llm/generate_method2.py`

Có thể chạy lại các script cũ để so sánh với pipeline mới ở `src/pipeline/*`.


## J. GENIE-style (theo Lin et al., ICML 2023)

Triển khai mới theo 3 khối:
- `src/dlm/genie_core.py`: diffusion schedule + q/p sampler
- `src/dlm/genie_denoiser.py`: timestep-conditioned denoiser với cross-attention lên `H_s`
- `src/dlm/genie_grounding.py`: decoder/grounding latent -> text

### Train GENIE denoiser
```bash
python3 src/pipeline/train_genie.py --model google/flan-t5-base --train-split train --val-split validation --max-train-samples 3000 --max-val-samples 300 --timesteps 100 --epochs 10 --output-dir outputs/models/genie
```

### Infer PP1 với GENIE
```bash
python3 src/pipeline/infer_method1_genie.py --llm-model-dir outputs/models/llm_sft_3000_qlora --genie-model-dir outputs/models/genie --split test --max-samples 300 --output outputs/drafts/method1_genie_300.csv
```

### Evaluate
```bash
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_genie_300.csv
```


## K. Evidence-Driven Study Runner

Runner `src/experiment/run_study.py` đã được mở rộng để chạy đầy đủ các biến thể đề cương + runtime:
- baseline_llm
- method1_single
- method1_multi_refine_each
- method1_multi_aggregate_mean
- method1_multi_aggregate_learned
- method2_dlm_llm

Chạy 1 lệnh:
```bash
python3 src/experiment/run_study.py --train-models --max-train-samples 3000 --max-val-samples 300 --max-test-samples 300 --exp-name thesis_ablation_300
```

Tổng hợp bảng kết quả + thời gian:
```bash
python3 src/evaluation/aggregate_metrics.py --exp-dir outputs/experiments/<exp_id>
```


### Adaptive Step Budget (GENIE extension)

Train step controller:
```bash
python3 src/pipeline/train_step_controller.py --llm-model-dir outputs/models/llm_sft_3000_qlora --genie-model-dir outputs/models/genie --max-samples 3000 --step-bins "20,40,60,80,100" --output outputs/models/genie/step_controller.pt
```

Infer with adaptive steps:
```bash
python3 src/pipeline/infer_method1_genie.py --llm-model-dir outputs/models/llm_sft_3000_qlora --genie-model-dir outputs/models/genie --adaptive-steps --step-controller-model outputs/models/genie/step_controller.pt --split test --max-samples 300 --output outputs/drafts/method1_genie_adaptive_300.csv
```

Evaluate adaptive vs fixed:
```bash
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_genie_300.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_genie_adaptive_300.csv
```


## L. PP2 Latent Diffusion Thuần

Train PP2 latent model:
```bash
python3 src/pipeline/train_method2_latent.py --model google/flan-t5-base --train-split train --val-split validation --max-train-samples 3000 --max-val-samples 300 --timesteps 100 --epochs 10 --output-dir outputs/models/method2_latent
```

Infer PP2 latent:
```bash
python3 src/pipeline/infer_method2_latent.py --method2-model-dir outputs/models/method2_latent --split test --max-samples 300 --output outputs/drafts/method2_latent_300.csv
```

Evaluate:
```bash
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method2_latent_300.csv
```
