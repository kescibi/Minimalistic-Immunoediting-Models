# Prediction Performance (OAK Study)

Prediction performance for the patients presented analogously to Figure 7 in the paper. See the tables below for exact computed PD probabilities and confusion matrices.

<p align="center">
  <img src="./chosen_patient_1_OAK_predict_1.png">
  <img src="./chosen_patient_6_OAK_predict_1.png">
  <img src="./chosen_patient_11_OAK_predict_1.png">
  <img src="./chosen_patient_16_OAK_predict_1.png">
  <img src="./chosen_patient_21_OAK_predict_1.png">
  <img src="./chosen_patient_26_OAK_predict_1.png">
  <img src="./chosen_patient_31_OAK_predict_1.png">
  <img src="./chosen_patient_36_OAK_predict_1.png">
  <img src="./chosen_patient_41_OAK_predict_1.png">
  <img src="./chosen_patient_46_OAK_predict_1.png">
  <img src="./chosen_patient_51_OAK_predict_1.png">
  <img src="./chosen_patient_56_OAK_predict_1.png">
  <img src="./chosen_patient_61_OAK_predict_1.png">
  <img src="./chosen_patient_66_OAK_predict_1.png">
  <img src="./chosen_patient_71_OAK_predict_1.png">
  <img src="./chosen_patient_76_OAK_predict_1.png">
  <img src="./chosen_patient_81_OAK_predict_1.png">
  <img src="./chosen_patient_86_OAK_predict_1.png">
  <img src="./chosen_patient_91_OAK_predict_1.png">
  <img src="./chosen_patient_96_OAK_predict_1.png">
  <img src="./chosen_patient_101_OAK_predict_1.png">
  <img src="./chosen_patient_106_OAK_predict_1.png">
  <img src="./chosen_patient_111_OAK_predict_1.png">
  <img src="./chosen_patient_116_OAK_predict_1.png">
  <img src="./chosen_patient_121_OAK_predict_1.png">
  <img src="./chosen_patient_126_OAK_predict_1.png">
  <img src="./chosen_patient_131_OAK_predict_1.png">
  <img src="./chosen_patient_136_OAK_predict_1.png">
  <img src="./chosen_patient_141_OAK_predict_1.png">
  <img src="./chosen_patient_146_OAK_predict_1.png">
  <img src="./chosen_patient_151_OAK_predict_1.png">
</p>

In the table below, we report the following information for each patient seen above: the patient ID, the Ground Truth (GT) progressive disease outcome, the model-derived ensemble probability (rounded), the classification outcome, and whether the ensemble was fully certain of its outcome.

| ID | GT | Ensemble Probability | Classification Outcome | Fully Certain? |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 0.25 | TN | false |
| 2 | 1 | 0 | FN | true |
| 3 | 0 | 0.375 | TN | false |
| 4 | 1 | 1 | TP | true |
| 5 | 0 | 1 | FP | true |
| 6 | 0 | 0 | TN | true |
| 7 | 0 | 0.944 | FP | false |
| 8 | 0 | 0.273 | TN | false |
| 9 | 0 | 0.588 | FP | false |
| 10 | 1 | 1 | TP | true |
| 11 | 1 | 1 | TP | true |
| 12 | 0 | 0 | TN | true |
| 13 | 0 | 0.25 | TN | false |
| 14 | 1 | 1 | TP | true |
| 15 | 1 | 1 | TP | true |
| 16 | 1 | 1 | TP | true |
| 17 | 1 | 1 | TP | true |
| 18 | 0 | 0 | TN | true |
| 19 | 0 | 0 | TN | true |
| 20 | 0 | 0 | TN | true |
| 21 | 1 | 1 | TP | true |
| 22 | 0 | 0.375 | TN | false |
| 23 | 0 | 0 | TN | true |
| 24 | 1 | 1 | TP | true |
| 25 | 1 | 1 | TP | true |
| 26 | 1 | 0.75 | TP | false |
| 27 | 1 | 0.002 | FN | false |
| 28 | 1 | 0.037 | FN | false |
| 29 | 0 | 0.008 | TN | false |
| 30 | 0 | 0 | TN | true |
| 31 | 0 | 0 | TN | true |
| 32 | 0 | 0.393 | TN | false |
| 33 | 1 | 1 | TP | true |
| 34 | 0 | 0.315 | TN | false |
| 35 | 0 | 0.017 | TN | false |
| 36 | 1 | 0 | FN | true |
| 37 | 1 | 1 | TP | true |
| 38 | 0 | 0.199 | TN | false |
| 39 | 0 | 0.882 | FP | false |
| 40 | 1 | 0 | FN | true |
| 41 | 0 | 0.125 | TN | false |
| 42 | 1 | 1 | TP | true |
| 43 | 1 | 0.019 | FN | false |
| 44 | 1 | 0.993 | TP | false |
| 45 | 0 | 0.9 | FP | false |
| 46 | 1 | 0 | FN | true |
| 47 | 0 | 0.375 | TN | false |
| 48 | 1 | 1 | TP | true |
| 49 | 1 | 0.005 | FN | false |
| 50 | 1 | 1 | TP | true |
| 51 | 1 | 1 | TP | true |
| 52 | 1 | 0 | FN | true |
| 53 | 1 | 0 | FN | true |
| 54 | 0 | 1 | FP | true |
| 55 | 1 | 1 | TP | true |
| 56 | 0 | 0.13 | TN | false |
| 57 | 0 | 0.019 | TN | false |
| 58 | 0 | 1 | FP | true |
| 59 | 0 | 0.309 | TN | false |
| 60 | 0 | 0 | TN | true |
| 61 | 0 | 0.625 | FP | false |
| 62 | 0 | 0.375 | TN | false |
| 63 | 0 | 1 | FP | true |
| 64 | 0 | 0.375 | TN | false |
| 65 | 1 | 1 | TP | true |
| 66 | 0 | 0.25 | TN | false |
| 67 | 1 | 1 | TP | true |
| 68 | 0 | 0.001 | TN | false |
| 69 | 1 | 0.922 | TP | false |
| 70 | 1 | 0.144 | FN | false |
| 71 | 0 | 0.025 | TN | false |
| 72 | 1 | 1 | TP | true |
| 73 | 0 | 0.825 | FP | false |
| 74 | 1 | 0.925 | TP | false |
| 75 | 0 | 0.4 | TN | false |
| 76 | 1 | 1 | TP | true |
| 77 | 1 | 1 | TP | true |
| 78 | 1 | 0.794 | TP | false |
| 79 | 1 | 0 | FN | true |
| 80 | 1 | 1 | TP | true |
| 81 | 1 | 0.449 | FN | false |
| 82 | 1 | 0.419 | FN | false |
| 83 | 0 | 0 | TN | true |
| 84 | 0 | 0.107 | TN | false |
| 85 | 0 | 0.001 | TN | false |
| 86 | 0 | 0.002 | TN | false |
| 87 | 0 | 0 | TN | true |
| 88 | 1 | 0.307 | FN | false |
| 89 | 1 | 1 | TP | true |
| 90 | 1 | 0 | FN | true |
| 91 | 0 | 0.774 | FP | false |
| 92 | 0 | 0.285 | TN | false |
| 93 | 0 | 0.415 | TN | false |
| 94 | 0 | 1 | FP | true |
| 95 | 1 | 1 | TP | true |
| 96 | 0 | 0.469 | TN | false |
| 97 | 0 | 1 | FP | true |
| 98 | 1 | 1 | TP | true |
| 99 | 1 | 0.285 | FN | false |
| 100 | 1 | 0.001 | FN | false |
| 101 | 1 | 0.752 | TP | false |
| 102 | 1 | 0.2 | FN | false |
| 103 | 1 | 1 | TP | true |
| 104 | 1 | 0.249 | FN | false |
| 105 | 0 | 0.097 | TN | false |
| 106 | 1 | 1 | TP | true |
| 107 | 0 | 0 | TN | true |
| 108 | 1 | 1 | TP | true |
| 109 | 0 | 0 | TN | true |
| 110 | 1 | 0.05 | FN | false |
| 111 | 0 | 1 | FP | true |
| 112 | 0 | 0 | TN | false |
| 113 | 0 | 0.006 | TN | false |
| 114 | 1 | 1 | TP | true |
| 115 | 1 | 1 | TP | true |
| 116 | 0 | 1 | FP | true |
| 117 | 1 | 1 | TP | true |
| 118 | 0 | 0 | TN | true |
| 119 | 1 | 0.416 | FN | false |
| 120 | 1 | 0.06 | FN | false |
| 121 | 0 | 1 | FP | true |
| 122 | 0 | 0 | TN | true |
| 123 | 1 | 0.012 | FN | false |
| 124 | 0 | 0.766 | FP | false |
| 125 | 1 | 0.369 | FN | false |
| 126 | 1 | 0.973 | TP | false |
| 127 | 0 | 0 | TN | true |
| 128 | 1 | 1 | TP | true |
| 129 | 1 | 1 | TP | true |
| 130 | 0 | 0.381 | TN | false |
| 131 | 0 | 0 | TN | true |
| 132 | 1 | 1 | TP | true |
| 133 | 0 | 1 | FP | true |
| 134 | 1 | 1 | TP | true |
| 135 | 1 | 0.65 | TP | false |
| 136 | 0 | 0.598 | FP | false |
| 137 | 0 | 0.167 | TN | false |
| 138 | 1 | 0.001 | FN | false |
| 139 | 0 | 0.051 | TN | false |
| 140 | 0 | 0 | TN | true |
| 141 | 0 | 0.389 | TN | false |
| 142 | 0 | 1 | FP | true |
| 143 | 0 | 0 | TN | true |
| 144 | 1 | 1 | TP | true |
| 145 | 1 | 1 | TP | true |
| 146 | 1 | 1 | TP | true |
| 147 | 1 | 0 | FN | true |
| 148 | 1 | 1 | TP | true |
| 149 | 0 | 0.088 | TN | false |
| 150 | 0 | 0.225 | TN | false |
| 151 | 1 | 0.993 | TP | false |
| 152 | 1 | 1 | TP | true |
| 153 | 1 | 0.213 | FN | false |
| 154 | 0 | 0.375 | TN | false |
| 155 | 0 | 0.203 | TN | false |

With the above computations, we obtain the following confusion matrices (considering all patients, and only the ones with full certainty, respectively):

|   | P | N |
|---|----|----|
| PP | 50 | 20 |
| PN | 28 | 57 |

|   | P | N |
|---|----|----|
| PP | 41 | 11 |
| PN | 9 | 19 |
