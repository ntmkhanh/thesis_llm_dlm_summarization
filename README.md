# thesis_llm_dlm_summarization

Pipeline thực nghiệm tóm tắt văn bản theo đề cương luận văn:
- Baseline LLM
- Phương pháp 1: LLM draft -> refine
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
python3 src/llm/generate_baseline.py --split "test" --output outputs/drafts/baseline_qwen.csv
```

Method 1:
```bash
python3 src/llm/generate_method1.py --mode single --split "test" --output outputs/drafts/method1_single.csv
```

Method 2:
```bash
python3 src/pipeline/infer_method2_latent.py --method2-model-dir outputs/models/method2_latent --split "test" --output outputs/drafts/method2_latent.csv
```

Chạy theo đúng pipeline đề cương:

Baseline LLM:
```bash
python3 src/llm/generate_baseline.py --split "test" --output outputs/drafts/baseline_qwen.csv
```

Phương pháp 1 (LLM draft -> refine):
```bash
python3 src/llm/generate_method1.py --mode single --split "test" --output outputs/drafts/method1_single.csv
python3 src/llm/generate_method1.py --mode multi --multi-mode refine_each --num-candidates 6 --split "test" --output outputs/drafts/method1_multi_refine_each.csv
python3 src/llm/generate_method1.py --mode multi --multi-mode aggregate_then_refine --num-candidates 3 --split "test" --output outputs/drafts/method1_multi_aggregate.csv
```

Phương pháp 2 (Latent Diffusion Transformer):
```bash
python3 src/pipeline/train_method2_latent.py --model google/flan-t5-base --output-dir outputs/models/method2_latent
python3 src/pipeline/infer_method2_latent.py --method2-model-dir outputs/models/method2_latent --split "test" --output outputs/drafts/method2_latent.csv
```

## 3) Đánh giá

```bash
python3 src/evaluation/compute_metrics.py --input outputs/drafts/baseline_qwen.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_single.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method2_latent.csv
```

Nếu cần chạy nhanh để debug, bạn có thể thay `--split "test"` bằng `--split "test[:100]"`.

## Ghi chú
- Script đã lọc `cnn_only=True` trong loader.
- Lần chạy đầu sẽ tải model và dataset từ Hugging Face.
- Cần máy có GPU để chạy nhanh.
