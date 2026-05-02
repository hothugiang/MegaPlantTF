# MegaPlantTF

Project gốc được clone từ:

```text
https://github.com/Bioinformatics-UM6P/MegaPlantTF
```

## Cách chạy nhanh

Tạo môi trường:

```powershell
cd MegaPlantTF
conda env create -f MegaPlantTF.yml
conda activate MegaPlantTF
python -m ipykernel install --user --name MegaPlantTF --display-name "MegaPlantTF"
```

Mở Jupyter:

```powershell
jupyter-lab
```

Notebook suy luận bằng mô hình pretrained:

```text
test/1-Start-With-MegaPlantTF.ipynb
```

Train FNN gốc cho 1 family:

```powershell
python notebook/01-approach2_kmer_neural_network.py AP2
```

Train nhiều family bằng pipeline notebook:

```powershell
cd notebook
python pyrunner.py
```

## Cấu trúc thư mục chính

```text
data/                  dữ liệu train/test và metadata
models/                mô hình pretrained và train reports
notebook/              notebook và script train/benchmark
pretrained/            code predictor để suy luận
processing/            hàm xử lý dữ liệu và tạo đặc trưng
test/metrics/          nơi lưu kết quả thực nghiệm
benchmark.qmd          mô tả benchmark chi tiết
guide.md               ghi chú và hướng dẫn bổ sung
```

## Thực nghiệm bổ sung `03-classifier_benchmark.py`

File:

```text
notebook/03-classifier_benchmark.py
```

Script này là phần bổ sung để benchmark các mô hình phân loại trên bài toán binary theo từng họ TF. Pipeline chung:

1. Đọc split train/test.
2. Chuyển `sequence` thành vector `k`-mer frequency.
3. Vector hóa bằng `DictVectorizer`.
4. Chọn đặc trưng bằng `SelectKBest(f_classif)`.
5. Train và evaluate mô hình trên cùng một protocol.

Các mô hình hỗ trợ:

```text
knn
xgb
random_forest
adaboost
svc_linear
svc_rbf
fnn_pretrained
catboost
```

Lưu ý:

- `fnn_pretrained` là mô hình FNN đã train sẵn trong `models/Binary-Classifier/`, chỉ load để evaluate.
- `catboost` là mô hình **bổ sung ngoài bài báo**, được thêm để mở rộng so sánh với một boosting model mạnh trên dữ liệu tabular.

Thông tin mô tả mô hình và hyperparameter nằm trong:

```text
benchmark.qmd
benchmark.pdf
```

## Cách chạy benchmark

Chạy benchmark theo split của paper:

```powershell
python -u notebook/03-classifier_benchmark.py --all-families --data-mode paper_split --k 3 --models knn,xgb,random_forest,adaboost,svc_linear,svc_rbf,fnn_pretrained --n-jobs 1
```

Nếu muốn dùng `catboost`, cần cài thêm package:

```powershell
pip install catboost
```

Chạy riêng `catboost` và không ghi đè file cũ:

```powershell
python -u notebook/03-classifier_benchmark.py --all-families --data-mode paper_split --k 3 --models catboost --max-rows 1000 --n-jobs 1 --output test/metrics/model_benchmark/catboost_benchmark.csv --summary-output test/metrics/model_benchmark/catboost_benchmark_summary.csv --comparison-output test/metrics/model_benchmark/catboost_benchmark_comparison.csv
```

## Đường dẫn kết quả thực nghiệm

Kết quả benchmark mặc định:

```text
test/metrics/model_benchmark/binary_classifier_benchmark.csv
test/metrics/model_benchmark/binary_classifier_benchmark_summary.csv
```

Kết quả nếu chạy riêng CatBoost với tên file riêng:

```text
test/metrics/model_benchmark/catboost_benchmark.csv
test/metrics/model_benchmark/catboost_benchmark_summary.csv
```
