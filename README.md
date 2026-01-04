Imbalance-Aware Binary Classification

Built an end-to-end **binary classification pipeline** on an **anonymized tabular dataset** and evaluated it using **Balanced Error Rate (BER)** to handle class imbalance. The solution emphasizes **leakage-free evaluation** through **nested cross-validation**, **feature selection inside folds**, and **threshold tuning** based strictly on validation data.

<p align="center">
<img width="400" height="800" alt="image" src="https://github.com/user-attachments/assets/ca47bd59-7665-43bc-b353-abc351d8f367" />
</p>


---

## Dataset (Anonymized)

- **Rows:** 10,000 labeled samples  
- **Features:** 21 predictors (`x1` … `x21`)  
- **Target:** binary label (`0/1`)  
- **Class imbalance:**
  - Class 0: **7,503**
  - Class 1: **2,497** (≈ **3:1** imbalance)

---

## Metric Choice

Accuracy can be misleading under class imbalance. The primary metric used is:

- **Balanced Error Rate (BER)** — equally penalizes errors from both classes.

---

## Pipeline (Leakage-Safe)

### 1) Preprocessing (ColumnTransformer)

- **Categorical:** `x2, x3, x4` → most-frequent imputation + one-hot encoding  
- **Numerical:** `x15–x21` → median imputation + standard scaling  
- **Binary:** all remaining predictors → most-frequent imputation + passthrough  

### 2) Feature Selection

- **RFE (Recursive Feature Elimination)** applied **after preprocessing**
- Fit **only on training folds**, then applied to validation/test folds
- Reduced to **12 selected features** (post-transformation)

### 3) Class Imbalance Handling

- **SMOTE** applied **only on training folds**, and **only after RFE**  
  *(prevents leakage from synthetic sampling)*

### 4) Models Evaluated

- **Optimized MLP (final):** 2 hidden layers (SELU + dropout)
- **Enhanced MLP:** residual architecture variant
- **XGBoost + Random Forest ensemble**
- **SVM (RBF kernel)** with proper nested CV

---

## Evaluation Strategy (Unbiased)

Used **nested cross-validation** to avoid optimistic bias:

- **Outer CV:** 5-fold stratified (unbiased performance estimate)
- **Inner CV:** 3-fold stratified (model + threshold selection)

Additionally, standard **5-fold CV** was used to generate out-of-fold predictions for model comparison.

### ROC Curve (5-Fold Cross-Validation)
<p align="center">
<img width="536" height="547" alt="image" src="https://github.com/user-attachments/assets/ac81fca4-4179-4a69-87b1-584246ca53cf" />
</p>

---

## Results

- **Train (5-fold CV) BER:** **0.3368**
- **Test (nested CV) BER:** **0.3389** (mean across outer folds)  
  → Small gap indicates limited overfitting

**Comparisons:**
- **XGBoost + RF ensemble (nested CV BER):** ~**0.3396** (similar, higher complexity)
- **SVM (RBF) + RFE (nested CV BER):** ~**0.3461** (worse + slower)
- **AUC:** ~**0.72** (moderate separability)

**Final decision threshold:** **0.48** (tuned via CV to minimize BER)

### BER Comparison (5-Fold CV)
<p align="center">
<img width="545" height="374" alt="image" src="https://github.com/user-attachments/assets/a1a7f0ca-b6dd-43a0-9bad-7a0284094e4e" />
</p>

### AUC Comparison (5-Fold CV)
<p align="center">
<img width="536" height="374" alt="image" src="https://github.com/user-attachments/assets/d3b78652-6688-4df4-9e36-1e459e2cef4f" />
</p>

---

## Final Inference on Unlabeled Set

Retrained the selected pipeline on **all 10,000 labeled samples** and applied it to the unlabeled dataset using the **BER-optimal threshold (0.48)**.

- **~44.9% positives**
- **~55.1% negatives**

---

## Interpretation & Takeaways

- PCA shows strong class overlap → task is **moderately difficult**
- Best performance came from **non-linear modeling + threshold tuning**
- Nested CV ensured feature selection, SMOTE, and threshold tuning stayed **inside training folds** (no leakage)

### PCA Projection (Class Overlap)

<img width="844" height="701" alt="image" src="https://github.com/user-attachments/assets/0c0e14ee-3e74-4ce3-8559-7b4f3649f056" />


---



