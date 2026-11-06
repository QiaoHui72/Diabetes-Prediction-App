# AI Diabetes Risk Assessment

Run from this extracted folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep all Python files, trained_models.pkl and diabetes.csv together. The app loads the included tuned models and evaluation results instead of training at startup. Requirements are pinned to the versions used to create the saved models. Use Python 3.12. The pair plot is generated only when Show pair plot is checked; other chart images are cached.

## Interface

- Predict: the blue reference design, calculated or manually entered BMI, selected model, prediction and input summary.
- Dataset: data table, summary, missing measurements, outcome distribution, annotated correlation heatmap, and a selectable 2–5-feature pair plot.
- Models: original versus expanded search CV scores, held-out accuracy, precision, recall, F1, ROC AUC, comparison chart, labelled confusion matrices, ROC curves, selected settings and CSV/model downloads.

Chart colours: blue = no diabetes; orange = diabetes. Heatmap values can differ from the example slides because invalid zero measurements are treated as missing rather than real measurements. Pair-plot panels omit missing measurements. Correlations use pairwise available data.

## Tuning and evaluation

Models use the original four inputs (Glucose, BloodPressure, BMI, Age). The additional dataset columns are available for exploration but are not silently substituted into prediction.

The same stratified 80/20 split with seed 42 reserves 154 records for testing. Searches use five shuffled stratified folds (seed 42) on the other 614 records. Median imputation and scaling are fitted inside each fold to prevent preprocessing leakage.

The original search spaces are rerun on these same folds as the baseline. Expanded searches explore additional neighbours, distance metrics, regularization, tree sizes and pruning; suitable models also compare standard and robust scaling. SVM and gradient boosting are additional candidates. Each original model retains its baseline if the expanded search scores lower. All model and setting choices are based on training-fold accuracy. The held-out test is evaluated only after choices are fixed; it does not choose the winner. The probability threshold remains 0.5.

The chosen CV score is a tuning score, not an unbiased estimate of future performance. Standard deviation describes variation across folds, not a confidence interval. Small improvements may reflect sampling variation. More tuning cannot guarantee better accuracy on new records. Accuracy should be considered with recall and false negatives, especially for this imbalanced dataset. Model probabilities are uncalibrated estimates, not verified clinical risks.

`evaluation_results.csv` records results from the verification run; installed library versions can affect retrained results. The app always computes its displayed scores from its actual models.

This is a learning project, not medical advice.

For the exact versions used during verification, install `requirements-tested.txt` instead of `requirements.txt`.

## Update your existing Streamlit deployment

Upload the extracted contents of this folder to the same GitHub repository, replacing existing files. Include trained_models.pkl and the updated requirements.txt. Do not upload just the ZIP. Keep app.py as the entrypoint. After dependencies are installed, the app loads the saved models without a tuning search.

For intentional model updates, run `python build_models.py` locally and upload the resulting trained_models.pkl together with diabetes.csv and matching requirements.txt. The app checks the dataset hash and package versions to avoid displaying mismatched scores. Only load model files you trust.
