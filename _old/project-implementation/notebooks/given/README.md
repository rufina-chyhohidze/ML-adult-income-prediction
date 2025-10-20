# Fixed Simple Adult Income ML Project

This project is **simple, readable, and fully commented**. It follows your project description:
1. Data understanding & EDA
2. Clean preprocessing (no leakage)
3. Two baseline models (Logistic Regression, Random Forest)
4. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC) + plots
5. Permutation importance
6. Save final model

## How to run
```bash
pip install -r requirements.txt
jupyter lab
# open notebooks/01_AdultIncome_Simple_Lifecycle.ipynb and Run All
```

### Data
The notebook expects your Adult dataset in `./data/`. I already extracted any uploaded `/mnt/data/adult.zip` here.
If needed, set `DATA_PATH` inside the notebook to the exact filename (e.g., `data/adult.data` or `data/adult.csv`).

## Notes
- Works across scikit-learn versions (no deprecated OneHotEncoder args).
- ColumnTransformer returns **dense** output to avoid sparse/dense surprises.
- ROC plots specify `pos_label=">50K"` for string targets.
- All key steps are commented with plain language.
