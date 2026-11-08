# Mobile Price Range Classification

## a. Problem statement
Predict the price range (0 = lowest, 3 = highest) of a mobile phone from its
hardware specifications. Five classification models are trained on the same
dataset, compared on six evaluation metrics, and demonstrated in an interactive
Streamlit web app.

## b. Dataset description
- **Source:** Kaggle - Mobile Price Classification
  (https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- **Instances:** 2000 rows (1600 for training, 400 for testing, 80/20 stratified split)
- **Features:** 20 numeric features, such as battery_power, ram, px_height,
  px_width, int_memory, mobile_wt, n_cores, and binary flags (blue, wifi, dual_sim, four_g, three_g, touch_screen)
- **Target:** price_range, a multi-class label with 4 balanced classes (0, 1, 2, 3)
- **Preprocessing:** no missing values; features standardized with StandardScaler
  inside a scikit-learn Pipeline. The pipeline is saved with each model.

## c. GitHub repository link
https://github.com/mg21212050-source/ml-assign2-2026AK05268/new/main)

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| KNN | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Support Vector Machine | 0.8950 | 0.9833 | 0.8969 | 0.8950 | 0.8956 | 0.8602 |

Precision, Recall and F1 are weighted averages over the 4 classes; AUC is one-vs-rest weighted.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on every metric (accuracy 0.965, MCC 0.953), which suggests the classes are almost linearly separable in the scaled feature space. |
| Decision Tree | Mid-range results (accuracy 0.83); a single tree makes hard, axis-aligned splits and is likely overfitting the training data. |
| KNN | Weakest model (accuracy 0.50, MCC 0.335); with 20 features, many of them not very informative, distances become less meaningful. |
| Naive Bayes | Accuracy 0.81 but a high AUC (0.95), so its probability ranking is good even where its class predictions are less accurate; the feature independence assumption is only partly true here. |
| Support Vector Machine | Second best (accuracy 0.895, MCC 0.860); handles the multi-class problem well but did not match the linear model. |

**Overall winner:** Logistic Regression. It has the highest value on all six metrics
(Accuracy 0.9650, AUC 0.9987, MCC 0.9534).

## Streamlit app
Live app: <<paste Streamlit link here>>
Features: CSV upload, model dropdown, evaluation metrics, confusion matrix and classification report.
The app loads the saved models and does not retrain.

## Repository contents
`app.py`, `train.py` (or notebook), `requirements.txt`, `test_data.csv`, `model/` (saved models)
