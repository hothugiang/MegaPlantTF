# Hướng dẫn tái lập MegaPlantTF

Tài liệu này ánh xạ các thí nghiệm chính trong paper **MegaPlantTF: a machine learning framework for comprehensive identification and classification of plant transcription factors** sang các file, notebook và script có trong repo này.

Paper chính thức: <https://doi.org/10.1093/bioinformatics/btaf678>

Repo hiện hỗ trợ 4 luồng chạy thực tế:

1. Huấn luyện các mô hình nhị phân theo từng họ TF và từng giá trị `k`.
2. Huấn luyện và đánh giá meta-classifier của tầng 2.
3. Tái chạy so sánh baseline với BLAST.
4. Chạy phân tích Sorghum / phân tích liên loài dùng trong các hình của paper.

## 0. Paper tập trung vào gì

Paper mô tả MegaPlantTF như một framework học máy để nhận diện protein có phải transcription factor (TF) hay không và phân loại TF theo họ.

Các điểm chính cần tái lập:

- dữ liệu từ PlantTFDB v4.0, gồm 320,370 TF protein sequence thuộc 58 họ;
- biểu diễn protein bằng tần suất k-mer với `k = 2, 3, 4, 5`;
- chọn đặc trưng bằng ANOVA;
- tầng 1 gồm 58 binary feed-forward neural network, mỗi mô hình ứng với một họ TF;
- tầng 2 dùng stacking meta-classifier, ngoài ra có baseline `Max Voting`;
- so sánh với BLAST trên test set 20%;
- đánh giá theo các threshold, nổi bật là `0.0`, `0.5`, `0.95`;
- case study genome-wide trên *Sorghum bicolor* và phân tích liên loài.

Trong repo, các phần này nằm chủ yếu ở:

- `notebook/01-approach2_kmer_neural_network.py`
- `notebook/01-approach2_kmer_neural_network.ipynb`
- `notebook/02-meta_classifier.py`
- `notebook/02-meta_classifier.ipynb`
- `test/k2_evaluation_models.ipynb`
- `test/k3_evaluation_models.ipynb`
- `test/k4_evaluation_models.ipynb`
- `test/k5_evaluation_models.ipynb`
- `test/comparison_with_blast.ipynb`
- `analysis/phylogenic-analysis1.ipynb`
- `analysis/phylogenic-analysis2.ipynb`

## 1. Thiết lập môi trường

Repo dùng Conda environment trong `MegaPlantTF.yml`.

```bash
conda env create -f MegaPlantTF.yml
conda activate MegaPlantTF
python -m ipykernel install --user --name MegaPlantTF --display-name "MegaPlantTF"
```

Nếu chạy notebook, mở Jupyter từ thư mục gốc của repo:

```bash
jupyter-lab
```

## 2. Tải pretrained weights

Các notebook inference và evaluation cần model weights nằm trong:

- `models/Binary-Classifier`
- `models/MetaClassifier`

Nếu hai thư mục này chưa có, tải từ Hugging Face repo được nhắc trong `README.md` và `models/README.md`:

```bash
git lfs install
git lfs clone https://huggingface.co/Genereux-akotenou/genomics-tf-prediction
```

Sau đó copy:

- `Binary-Classifier/` vào `models/Binary-Classifier/`
- `MetaClassifier/` vào `models/MetaClassifier/`

Code dự đoán local sẽ load:

- binary model từ `models/Binary-Classifier/<family_code>/FEEDFORWARD_k{k}.keras`
- meta model từ `models/MetaClassifier/META_k{k}.keras`

## 3. Chạy suy luận nhanh với pretrained model

Cách nhanh nhất là dùng script runner:

```bash
cd run-as-script
python run_megaplanttf.py --fasta __temp__/Ach_pep_kiwi-small.fas --kmer 3 --voting "Two-Stage Voting" --output output
```

Script sẽ sinh:

- `MegaPlantTF_Dashboard_<jobid>.html`
- `MegaPlantTF_Dashboard_<jobid>_predictions.csv`

Ghi chú:

- `--kmer 3` là cấu hình nên dùng để demo nhanh vì paper báo cáo `k = 3` cùng threshold `0.5` là cấu hình nổi bật.
- `--voting` nhận `"Two-Stage Voting"` hoặc `"Max Voting"`.
- `--threshold` chủ yếu phục vụ lọc/hiển thị kết quả theo confidence threshold.
- Với FASTA lớn, nên dùng batch predictor trong `pretrained/predictor.py`, đặc biệt class `BatchSingleKModel`.

## 4. Tái lập binary classifier theo từng họ TF

Binary classifiers được huấn luyện bằng:

- `notebook/01-approach2_kmer_neural_network.py`
- `notebook/01-approach2_kmer_neural_network.ipynb`
- `notebook/pyrunner.py`

Các input chính:

- `data/gene_info.json`
- `data/gene_info_small.json`
- `data/one_vs_other/<family>.csv`

### 4.1 Chạy thử trên tập nhỏ

Hiện tại `notebook/pyrunner.py` đang trỏ tới:

```python
gene_info_path = "../data/gene_info_small.json"
```

Chạy từ thư mục `notebook/`:

```bash
cd notebook
python pyrunner.py
```

Runner sẽ dùng Papermill để chạy `01-approach2_kmer_neural_network.ipynb` cho từng họ trong file JSON đã chọn và lưu notebook output vào:

- `notebook/AutoSave/`

Trong bản hiện tại, runner dùng kernel `python3` và `multiprocess = True` với tối đa 5 process.

### 4.2 Chạy quy mô đầy đủ như paper

Để chạy toàn bộ họ TF, đổi trong `notebook/pyrunner.py`:

```python
gene_info_path = "../data/gene_info_small.json"
```

thành:

```python
gene_info_path = "../data/gene_info.json"
```

Sau đó chạy lại:

```bash
python pyrunner.py
```

Việc này tốn thời gian và tài nguyên đáng kể vì paper huấn luyện 58 binary classifiers cho nhiều giá trị `k`.

### 4.3 Artifact được sinh ra

Notebook huấn luyện ghi model vào:

- `notebook/Output/Model/<family_code>/FEEDFORWARD_k2.keras`
- `notebook/Output/Model/<family_code>/FEEDFORWARD_k3.keras`
- `notebook/Output/Model/<family_code>/FEEDFORWARD_k4.keras`
- `notebook/Output/Model/<family_code>/FEEDFORWARD_k5.keras`
- `notebook/Output/Model/<family_code>/meta.json`

Training report theo từng họ nằm ở:

- `models/Train-Reports/<family>/`

## 5. Huấn luyện meta-classifier tầng 2

Meta-classifier nhận đầu ra của các binary classifier và học cách gộp chúng thành dự đoán họ TF cuối cùng.

Các file liên quan:

- `notebook/02-meta_classifier.py`
- `notebook/02-meta_classifier.ipynb`
- `notebook/002-meta_classifier-Copy1.ipynb`

Input cần có:

- `data/gene_info.json`
- `data/raw_data/`
- binary models đã huấn luyện trong `notebook/Output/Model/`

Output mong đợi:

- `notebook/Output/MetaClassifier/META_k2.keras`
- `notebook/Output/MetaClassifier/META_k3.keras`
- `notebook/Output/MetaClassifier/META_k4.keras`
- `notebook/Output/MetaClassifier/META_k5.keras`
- `notebook/Output/MetaClassifier/META_k2.joblib`
- `notebook/Output/MetaClassifier/META_k3.joblib`
- `notebook/Output/MetaClassifier/META_k4.joblib`
- `notebook/Output/MetaClassifier/META_k5.joblib`

Khi chạy thủ công, giữ working directory là `notebook/` vì nhiều đường dẫn trong notebook là relative path.

## 6. Tái lập evaluation theo threshold

Paper so sánh BLAST, `Max Voting` và `Two-Stage Voting` qua nhiều giá trị `k` và threshold.

Notebook chính:

- `test/k2_evaluation_models.ipynb`
- `test/k3_evaluation_models.ipynb`
- `test/k4_evaluation_models.ipynb`
- `test/k5_evaluation_models.ipynb`

Ví dụ với `k = 5`, notebook dùng:

- `data/testset-full/k5/testset.csv`
- `data/testset-full/k5/true_labels.csv`
- `data/testset-full/k5/class_mapping.json`

Các lệnh đánh giá quan trọng dùng `GenBoard` từ `pretrained/histogram.py`:

```python
genboard.show_eval_metric(
    true_label=true_label,
    class_mapping_rules=class_mapping,
    voting_method="Two-Stage Voting",
    voting_threshold=0.5,
    binary_class_threshold=0.5,
)
```

```python
genboard.compute_and_save_roc_per_family(
    true_label=true_label,
    class_mapping_rules=class_mapping,
    voting_method="Max Voting",
    voting_threshold=0.1,
    save_folder="results/maxvoting/k3",
)
```

Output evaluation đã có sẵn trong repo:

- `test/metrics/method_two-stage/<threshold>/overall_classifier_<k>.json`
- `test/metrics/method_maxvoting/<threshold>/overall_classifier_<k>.json`
- `test/metrics/binary_classifier_<k>.json`
- `test/results/2ndvoting/k<k>/roc_data_k<k>.json`
- `test/results/maxvoting/k<k>/roc_data_k<k>.json`
- `test/report/k<k>_evaluation_models.pdf`

Các file này phù hợp để làm bảng hoặc biểu đồ trong slide.

## 7. Tái lập baseline BLAST

Baseline BLAST nằm ở:

- `test/comparison_with_blast.ipynb`
- `test/BLAST/utils/metric.py`

Notebook thực hiện:

1. Đọc `test/BLAST/src/testset.csv`.
2. Ghi FASTA cho test set và train set.
3. Tạo local BLAST database protein.
4. Chạy `blastp`.
5. So sánh dự đoán BLAST với nhãn thật.

Lệnh BLAST chính:

```bash
makeblastdb -in ./BLAST/trainset.fasta -dbtype prot -out ./BLAST/database/pygenomics_ref_db
blastp -query ./BLAST/testset.fasta -db ./BLAST/database/pygenomics_ref_db -out ./BLAST/blast_result.txt -outfmt 6 -evalue 1e-5 -num_threads 10
```

Output BLAST:

- `test/results/BLAST/blast_predictions.csv`
- `test/results/BLAST/blast_family_metrics_id80.json`
- `test/metrics/method_blast/blast_results.json`

Nếu máy chưa cài BLAST+, notebook sẽ dừng tại cell gọi `makeblastdb` hoặc `blastp`.

## 8. Tái lập phân tích Sorghum và liên loài

Paper có case study trên *Sorghum bicolor* và đánh giá khả năng tổng quát hóa theo loài.

Các notebook và dữ liệu liên quan:

- `analysis/phylogenic-analysis1.ipynb`
- `analysis/phylogenic-analysis2.ipynb`
- `analysis/by-species-megaplantf.ipynb`
- `analysis/by-species-blast.ipynb`
- `analysis/by-species-merged.ipynb`
- `analysis/data_megaplantf.csv`
- `analysis/data_blast.csv`
- `analysis/results/megaPlantTF_phylo.csv`
- `analysis/results/BLAST_phylo.csv`
- `analysis/results/megaPlantTF_phylo.json`
- `analysis/results/BLAST_phylo.json`

Các hình đã sinh sẵn:

- `analysis/phylogenetic_analysis_comparison.png`
- `analysis/species_accuracy_comparison.png`
- `analysis/results/blast_vs_megaplanttf_phylo.png`
- `analysis/results/MegaPlantTF_Radar_Comparison.png`

## 9. Gợi ý trình bày trên slide

Nếu mục tiêu là trình bày project theo paper, mạch nội dung nên là:

1. Bài toán: nhận diện và phân loại plant TF ở quy mô genome.
2. Dữ liệu: PlantTFDB v4.0, 58 họ, split train/test 80/20.
3. Pipeline: FASTA protein sequence -> k-mer frequency -> ANOVA feature selection -> 58 binary classifiers -> voting hoặc stacking.
4. Benchmark: BLAST so với `Max Voting` và `Two-Stage Voting`.
5. Kết quả chính: `k = 3`, threshold `0.5`, `Two-Stage Voting`.
6. Case study: *Sorghum bicolor* và phân tích liên loài.

Nếu cần demo nhanh, cấu hình nên dùng:

- `k = 3`
- `threshold = 0.5`
- `voting = "Two-Stage Voting"`

## 10. Ghi chú thực tế

- Chạy notebook từ đúng thư mục của nó vì repo dùng nhiều relative path.
- Trước khi inference, cần có `models/Binary-Classifier` và `models/MetaClassifier`.
- `MultiKModel` trong `pretrained/predictor.py` hiện vẫn là stub; đường chạy ổn định hơn là predictor cho một giá trị `k`.
- Trên Windows, các bước dùng `rsync`, `mktemp`, `makeblastdb` và `blastp` nên chạy trong Git Bash, WSL hoặc môi trường có BLAST+.
- Nếu chỉ cần tái hiện kết quả để báo cáo, ưu tiên dùng các JSON/PNG/PDF đã sinh trong `test/metrics/`, `test/results/`, `test/report/` và `analysis/results/`.
