# Fine-tune Qwen on CNN/DailyMail for Draft Summary

Thư mục này chỉ phục vụ bước đầu của Method 1:

```text
CNN/DailyMail
-> fine-tune Qwen
-> Qwen sinh draft summary
-> draft CSV dùng làm input cho DiffuSeq
```

Model mặc định:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Dataset:

```text
cnn_dailymail 3.0.0
```

Token config đang dùng:

```text
article / input: max_source_length = 1024
summary / highlights: max_target_length = 80
draft generation: max_new_tokens = 80
```

Chỉ lấy phần CNN trong dataset. Code loader dùng split gốc `train`, `validation`, `test`
của CNN/DailyMail rồi lọc các article bắt đầu bằng `(CNN)`, nên DailyMail bị loại.

Ý tưởng tương đương:

```python
train_data = dataset["train"].filter(lambda x: x["article"].lstrip().startswith("(CNN)"))
val_data = dataset["validation"].filter(lambda x: x["article"].lstrip().startswith("(CNN)"))
test_data = dataset["test"].filter(lambda x: x["article"].lstrip().startswith("(CNN)"))
```

Không dùng `.select(range(500))`, `.select(range(100))`, `.select(range(50))` trong cấu hình mặc định.
`0` nghĩa là lấy hết CNN của split đó.

Kiểm tra nhanh số lượng mẫu CNN-only:

```bash
python finetuning/qwen_cnn_dailymail/verify_cnn_only.py --split train
python finetuning/qwen_cnn_dailymail/verify_cnn_only.py --split validation
python finetuning/qwen_cnn_dailymail/verify_cnn_only.py --split test
```

## 1) Fine-tune Qwen

Mặc định script dùng LoRA để nhẹ hơn full fine-tuning:

```bash
python finetuning/qwen_cnn_dailymail/train.py \
  --max-train-samples 0 \
  --max-val-samples 0 \
  --max-source-length 1024 \
  --max-target-length 80 \
  --epochs 1 \
  --tuning-mode lora
```

Output mặc định:

```text
finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft
```

Trong thư mục này, Hugging Face `Trainer` sẽ lưu checkpoint theo epoch, ví dụ:

```text
finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft/checkpoint-*
```

Trainer sẽ chọn checkpoint tốt nhất theo `eval_loss`. Sau khi train xong, best model/tokenizer
được lưu ngay tại:

```text
finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft
```

Thông tin checkpoint tốt nhất được ghi ở:

```text
finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft/best_checkpoint.json
```

Loss history được lưu cùng thư mục:

```text
finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft/trainer_log_history.json
finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft/train_loss_history.json
finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft/eval_loss_history.json
```

Nếu muốn full fine-tuning:

```bash
python finetuning/qwen_cnn_dailymail/train.py --tuning-mode full
```

## 2) Sinh Draft Summary

Sinh draft cho train/validation/test:

```bash
python finetuning/qwen_cnn_dailymail/generate_drafts.py \
  --model-dir finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft \
  --split train \
  --max-samples 0 \
  --max-new-tokens 80

python finetuning/qwen_cnn_dailymail/generate_drafts.py \
  --model-dir finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft \
  --split validation \
  --max-samples 0 \
  --max-new-tokens 80

python finetuning/qwen_cnn_dailymail/generate_drafts.py \
  --model-dir finetuning/qwen_cnn_dailymail/checkpoints/qwen_cnn_dailymail_sft \
  --split test \
  --max-samples 0 \
  --max-new-tokens 80
```

Output mặc định:

```text
outputs/drafts/qwen_cnn_dailymail/qwen_train_drafts.csv
outputs/drafts/qwen_cnn_dailymail/qwen_validation_drafts.csv
outputs/drafts/qwen_cnn_dailymail/qwen_test_drafts.csv
```

## 3) Chạy Trọn Bước Qwen

```bash
python finetuning/qwen_cnn_dailymail/run_all.py \
  --max-train-samples 0 \
  --max-val-samples 0 \
  --max-test-samples 0 \
  --tuning-mode lora
```

Trong các lệnh trên, `0` nghĩa là lấy toàn bộ mẫu CNN-only của split gốc đó.

Sau bước này, dùng draft CSV để chuẩn bị data cho DiffuSeq:

```bash
python src/data/prepare_diffuseq_refine_data.py \
  --train-drafts outputs/drafts/qwen_cnn_dailymail/qwen_train_drafts.csv \
  --valid-drafts outputs/drafts/qwen_cnn_dailymail/qwen_validation_drafts.csv \
  --test-drafts outputs/drafts/qwen_cnn_dailymail/qwen_test_drafts.csv \
  --max-draft-words 80 \
  --max-reference-words 80 \
  --output-dir outputs/diffuseq_data/qwen_refine
```

Train DiffuSeq refiner:

```bash
python src/pipeline/train_diffuseq_refiner.py \
  --data-dir outputs/diffuseq_data/qwen_refine \
  --dataset qwen_cnn_refine \
  --learning-steps 50000 \
  --save-interval 10000 \
  --batch-size 64 \
  --microbatch 16 \
  --seq-len 512 \
  --notes qwen_cnn_refine
```

Decode refined summary trên test split:

```bash
python src/pipeline/infer_diffuseq_refiner.py \
  --model-dir external/DiffuSeq/diffusion_models/<TEN_THU_MUC_CHECKPOINT> \
  --split test \
  --output outputs/refined/qwen_diffuseq_test.csv
```
