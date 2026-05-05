# thesis_llm_dlm_summarization

Pipeline thực nghiệm tóm tắt văn bản theo đề cương luận văn:
- Baseline LLM
- Phương pháp 1: LLM draft -> refine
- Phương pháp 2: plan -> decode
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
python3 src/llm/generate_method2.py --split "test" --output outputs/drafts/method2_plan_decode.csv
```

## 3) Đánh giá

```bash
python3 src/evaluation/compute_metrics.py --input outputs/drafts/baseline_qwen.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method1_single.csv
python3 src/evaluation/compute_metrics.py --input outputs/drafts/method2_plan_decode.csv
```

## Ghi chú
- Script đã lọc `cnn_only=True` trong loader.
- Lần chạy đầu sẽ tải model và dataset từ Hugging Face.
- Cần máy có GPU để chạy nhanh.
