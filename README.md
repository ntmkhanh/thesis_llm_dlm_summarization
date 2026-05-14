# thesis_llm_dlm_summarization

Pipeline thực nghiệm tóm tắt văn bản theo đề cương luận văn:
- Baseline LLM
- Phương pháp 1: LLM draft -> DLM refine
- Phương pháp 2: latent diffusion transformer (article-conditioned)
- Đánh giá ROUGE + BERTScore

## 1) Clone và tạo môi trường

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd thesis_llm_dlm_summarization
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Chạy sinh tóm tắt (CNN only)

Baseline:
```bash
python3 src/llm/generate_baseline.py --split "test" --output outputs/drafts/baseline_llama.csv
```

Method 1 - Single Draft với DiffuSeq official:
```bash
python3 src/pipeline/train_llm_sft.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/models/qwen_sft

python3 src/pipeline/generate_llm_drafts.py \
  --llm-model-dir outputs/models/qwen_sft \
  --split test \
  --max-samples 100 \
  --output outputs/drafts/qwen_test_drafts.csv

# Sau đó chuẩn bị train/valid/test jsonl và fine-tune external/DiffuSeq như phần bên dưới.
```

Method 2:
```bash
python3 src/pipeline/infer_method2_latent.py --method2-model-dir outputs/models/method2_latent --split "test" --output outputs/drafts/method2_latent.csv
```

Chạy theo đúng pipeline đề cương:

Baseline LLM:
```bash
python3 src/llm/generate_baseline.py --split "test" --output outputs/drafts/baseline_llama.csv
```

Phương pháp 1 (LLM draft -> DiffuSeq refine):

DiffuSeq official được clone tại `external/DiffuSeq`.

Pipeline Method 1:

```text
CNN/DailyMail
↓
Fine-tune Qwen và LLaMA để sinh draft summary
↓
Draft summary + document
↓
DiffuSeq official refine
↓
Output summary
```

Hai LLM draft generators mặc định:
- `Qwen/Qwen2.5-1.5B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`

DiffuSeq refiner dùng code chính thức từ `Shark-NLP/DiffuSeq`, nhận `src` là
`document + draft summary` và `trg` là reference summary.

Sinh draft bằng LLM đã fine-tune:

```bash
python3 src/pipeline/generate_llm_drafts.py \
  --llm-model-dir outputs/models/qwen_sft \
  --split train \
  --max-samples 3000 \
  --output outputs/drafts/qwen_train_drafts.csv

python3 src/pipeline/generate_llm_drafts.py \
  --llm-model-dir outputs/models/qwen_sft \
  --split validation \
  --max-samples 300 \
  --output outputs/drafts/qwen_valid_drafts.csv

python3 src/pipeline/generate_llm_drafts.py \
  --llm-model-dir outputs/models/qwen_sft \
  --split test \
  --max-samples 100 \
  --output outputs/drafts/qwen_test_drafts.csv
```

Chuẩn bị data cho DiffuSeq fine-tuning:

```bash
python3 src/data/prepare_diffuseq_refine_data.py \
  --train-drafts outputs/drafts/qwen_train_drafts.csv \
  --valid-drafts outputs/drafts/qwen_valid_drafts.csv \
  --test-drafts outputs/drafts/qwen_test_drafts.csv \
  --output-dir outputs/diffuseq_data/qwen_refine
```

Fine-tune DiffuSeq official:

```bash
python3 src/pipeline/train_diffuseq_refiner.py \
  --data-dir outputs/diffuseq_data/qwen_refine \
  --dataset cnn_dailymail_qwen_refine \
  --nproc-per-node 1 \
  --diff-steps 2000 \
  --learning-steps 50000 \
  --save-interval 10000 \
  --batch-size 64 \
  --microbatch 16 \
  --seq-len 512 \
  --notes qwen_refine
```

Decode bằng DiffuSeq checkpoint đã fine-tune:

```bash
python3 src/pipeline/infer_diffuseq_refiner.py \
  --model-dir external/DiffuSeq/diffusion_models/<diffuseq-checkpoint-folder> \
  --split test \
  --step 2000 \
  --batch-size 16 \
  --output outputs/drafts/method1_qwen_diffuseq.csv
```

Lặp lại các lệnh trên với `outputs/models/llama_sft` để sinh/refine draft từ LLaMA.

Nếu muốn train DiffuSeq trên cả draft Qwen và LLaMA, truyền nhiều CSV cách nhau bằng dấu phẩy:

```bash
python3 src/data/prepare_diffuseq_refine_data.py \
  --train-drafts outputs/drafts/qwen_train_drafts.csv,outputs/drafts/llama_train_drafts.csv \
  --valid-drafts outputs/drafts/qwen_valid_drafts.csv,outputs/drafts/llama_valid_drafts.csv \
  --test-drafts outputs/drafts/qwen_test_drafts.csv \
  --output-dir outputs/diffuseq_data/qwen_llama_refine
```

Phương pháp 2 (Latent Diffusion Transformer):
```bash
python3 src/pipeline/train_method2_latent.py --model google/flan-t5-base --output-dir outputs/models/method2_latent
python3 src/pipeline/infer_method2_latent.py --method2-model-dir outputs/models/method2_latent --split "test" --output outputs/drafts/method2_latent.csv
```

## 3) Đánh giá

```bash
python3 src/evaluation/compute_metrics.py --input outputs/drafts/baseline_llama.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_single.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method2_latent.csv
```

Nếu cần chạy nhanh để debug pipeline trong `src/pipeline`, giữ `--split test` và thêm `--max-samples 100`.

`src/llm/generate_method1.py` là bản thử nghiệm LLM-only: LLM sinh draft rồi chính LLM sửa lại bằng prompt editor. Bản này không phải pipeline LLM + DLM đúng theo đề cương.

## Ghi chú
- Script đã lọc `cnn_only=True` trong loader.
- Lần chạy đầu sẽ tải model và dataset từ Hugging Face.
- LLaMA trên Hugging Face có thể cần đăng nhập và chấp nhận license của Meta trước khi tải model.
- Cần máy có GPU để chạy nhanh.
