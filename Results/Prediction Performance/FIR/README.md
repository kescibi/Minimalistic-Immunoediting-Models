# Prediction Performance (FIR Study)

Prediction performance for the patients presented analogously to Figure 7 in the paper. See the tables below for exact computed PD probabilities and confusion matrices.

<p align="center">
  <img src="./chosen_patient_1_FIR_predict_1.png">
  <img src="./chosen_patient_6_FIR_predict_1.png">
  <img src="./chosen_patient_11_FIR_predict_1.png">
  <img src="./chosen_patient_16_FIR_predict_1.png">
  <img src="./chosen_patient_21_FIR_predict_1.png">
  <img src="./chosen_patient_26_FIR_predict_1.png">
  <img src="./chosen_patient_31_FIR_predict_1.png">
</p>

In the table below, we report the following information for each patient seen above: the patient ID, the Ground Truth (GT) progressive disease outcome, the model-derived ensemble probability (rounded), the classification outcome, and whether the ensemble was fully certain of its outcome.

 ID | GT | Ensemble Probability | Classification Outcome | Fully Certain? |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 0.175 | TN | false |
| 2 | 0 | 0.332 | TN | false |
| 3 | 0 | 0.875 | FP | false |
| 4 | 0 | 0.465 | TN | false |
| 5 | 0 | 0 | TN | true |
| 6 | 0 | 0.012 | TN | false |
| 7 | 0 | 0.007 | TN | false |
| 8 | 0 | 0.707 | FP | false |
| 9 | 1 | 0.875 | TP | false |
| 10 | 0 | 0.002 | TN | false |
| 11 | 1 | 0.947 | TP | false |
| 12 | 0 | 0 | TN | true |
| 13 | 0 | 0.854 | FP | false |
| 14 | 0 | 0.989 | FP | false |
| 15 | 0 | 0.489 | TN | false |
| 16 | 1 | 1 | TP | true |
| 17 | 0 | 0 | TN | true |
| 18 | 0 | 0.5 | TN | false |
| 19 | 1 | 1 | TP | true |
| 20 | 0 | 0.016 | TN | false |
| 21 | 0 | 0.375 | TN | false |
| 22 | 0 | 0 | TN | true |
| 23 | 1 | 0.967 | TP | false |
| 24 | 0 | 0 | TN | false |
| 25 | 0 | 0.306 | TN | false |
| 26 | 0 | 0.179 | TN | false |
| 27 | 1 | 1 | TP | true |
| 28 | 0 | 0.375 | TN | false |
| 29 | 1 | 0.106 | FN | false |
| 30 | 0 | 0 | TN | true |
| 31 | 0 | 0 | TN | true |
| 32 | 0 | 0 | TN | true |
| 33 | 1 | 1 | TP | true |

With the above computations, we obtain the following confusion matrices (considering all patients, and only the ones with full certainty, respectively):

|   | P | N |
|---|----|----|
| PP | 7 | 4 |
| PN | 1 | 21 |

|   | P | N |
|---|----|----|
| PP | 4 | 0 |
| PN | 0 | 7 |
