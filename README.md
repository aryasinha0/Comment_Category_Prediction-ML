# 💬 Comment Category Prediction

A machine learning solution for the **Comment Category Prediction Challenge** on Kaggle, achieving an **R² score of 0.92** on the validation set and a **public Kaggle score of 0.807**.

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| Validation R² | **0.92** |
| Kaggle Public Score | **0.807** |
| Validation Accuracy | ~91.3% |

---

## 📌 Problem Statement

Given a dataset of user-generated comments from an online discussion platform, the goal is to predict the **final category (`label`) assigned to each comment** by the platform's internal system. The label can take **4 distinct values**, each representing a different internal handling category.

**Competition Timeline:** Jan 17, 2026 – Mar 31, 2026

---

## 📂 Dataset

| File | Description |
|------|-------------|
| `train.csv` | 198,000 records with features and target `label` |
| `test.csv` | 102,000 records with features only |
| `sample_submission.csv` | Submission format reference |

### Feature Description

| Feature | Description |
|---------|-------------|
| `comment` | Raw text content of the comment |
| `created_date` | Timestamp of when the comment was posted |
| `post_id` | Identifier linking the comment to a parent thread |
| `emoticon_1/2/3` | Indicators for three internal emoticon groups |
| `upvote` | Number of positive reactions |
| `downvote` | Number of negative reactions |
| `if_1`, `if_2` | Hidden internal platform features |
| `race` | Whether the system detected group-identity references |
| `religion` | Whether the system detected belief-related references |
| `gender` | Whether the system detected gender-related references |
| `disability` | Whether the system detected ability-related references |
| `label` | **Target** — final category assigned by the platform (0–3) |

---

## 🔧 Methodology

### 1. Exploratory Data Analysis
- Inspected class distribution, null values, and feature correlations
- `if_2` showed the highest correlation with `label` (0.23) among numerical features
- `race`, `religion`, and `gender` had ~73% missing values — treated as absence indicators
- `emoticon_3` showed notable correlation with `downvote` (0.37)

### 2. Feature Engineering

```python
# Identity indicators: NaN → absent (0), present (1)
df["race"]     = df["race"].notna().astype(int)
df["religion"] = df["religion"].notna().astype(int)
df["gender"]   = df["gender"].notna().astype(int)

# Temporal features from created_date
df["hour"]       = df["created_date"].dt.hour
df["dayofweek"]  = df["created_date"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# Engagement features
df["total_vote"]    = df["upvote"] + df["downvote"]
df["log_upvote"]    = np.log1p(df["upvote"])
df["log_downvote"]  = np.log1p(df["downvote"])
df["log_total_vote"] = np.log1p(df["total_vote"])
```

### 3. Text Preprocessing
- Applied a custom `normalize_text` function to clean comment text
- Used **TF-IDF vectorization** with unigrams + bigrams (`ngram_range=(1,2)`)
- Top 20,000 features, filtered with `min_df=5`, `max_df=0.9`, English stop words removed

### 4. Preprocessing Pipeline

```python
ct = ColumnTransformer(transformers=[
    ('scaler',    StandardScaler(with_mean=False), numeric_cols),
    ('min_abs',   MaxAbsScaler(),                  ['emoticon_1', 'emoticon_2', 'emoticon_3']),
    ('vectorizer', TfidfVectorizer(
        ngram_range=(1,2), max_features=20_000,
        min_df=5, max_df=0.9, stop_words='english'
    ), 'comment')
], remainder='passthrough')
```

### 5. Model

The final model is **LightGBM** (`LGBMClassifier`), chosen after experimenting with Logistic Regression, LinearSVC, SGDClassifier, XGBoost, Decision Tree, and MLP.

```python
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42,
    n_jobs=-1
)
```

**Train/Validation Split:** 80/20 stratified split (158,400 train / 39,600 val)  
**Features used:** 19,910 (after TF-IDF + structured features)

---

## 🧪 Models Explored

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline; various `C` values tried |
| LinearSVC | Grid searched over `C` |
| SGDClassifier | Grid searched over `alpha`, `eta0` |
| MLP Classifier | Configs: `(100,)`, `(128,64)`, `(256,128,64)` |
| XGBoost | Tried `n_estimators=200`, `max_depth=6` |
| **LightGBM** ✅ | **Final model** — best accuracy at ~91.3% |

---

## 📦 Tech Stack

- **Python 3.12**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, pipelines, evaluation
- `lightgbm` — final classifier
- `matplotlib`, `seaborn` — EDA visualizations
- Kaggle Notebooks (Papermill)

---

## 🗂 Project Structure

```
├── notebook.ipynb          # Full training notebook
├── submission.csv          # Final Kaggle submission
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install numpy pandas scikit-learn lightgbm matplotlib seaborn
   ```

2. Place `train.csv` and `test.csv` in the appropriate input path.

3. Run all cells in `notebook.ipynb` top to bottom.

4. The final `submission.csv` will be generated automatically.

---

## 📈 Submission Format

```csv
ID,label
1,2
2,0
3,1
...
```
