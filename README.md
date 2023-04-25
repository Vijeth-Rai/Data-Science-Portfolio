# Data-Science-Portfolio
A collection of my data science projects showcasing my skills in machine learning, artificial intelligence, and data analysis. Includes Jupyter notebook and visualizations demonstrating my expertise and problem-solving abilities.

# Contents

1. Microsoft Malware Detection
2. Stripper Well Problem
3. Steel Defect Detection
4. Student Mental Health - Outlier Detection
5. Instagram Follow Recommendation
6. Implementations from Scratch
    - TF-IDF
    - RandomSearchCV
    - SGD
    - Backpropagation
    - Linear Regression with Gradient Descent
    - KMeans Clustering
    - BERT
    - Seq2Seq
7. Master's Dissertation

# Microsoft Malware Detection 

### About the dataset
The dataset is taken from https://www.kaggle.com/c/microsoft-malware-prediction. This dataset contains .byte and .asm files which are most commonly used in malware detection tasks. The .byte files contain binary data in hexadecimal format and the .asm files contain the disassembled code of the binary files. There are 10868 .byte and .asm files in the dataset. The task is to classify whether the file classifies as a malware or not.

### Motivation
The motivation behind this project lies in the need to protect users from the ever-evolving landscape of cyber threats and ensure the safety and integrity of computer systems. By creating a reliable and efficient malware detection system using machine learning, we can stay one step ahead of cybercriminals and safeguard our digital lives.

### Preprocess
The .byte and .asm files have been transformed into uni-grams and bi-grams, capturing frequency and co-occurrence patterns in the data. Moreover, the binary representations of the .asm files have been converted into pixel values to analyze intensity patterns potentially indicative of malware presence. These feature representations facilitate improved understanding and classification by machine learning algorithms, ultimately enhancing the accuracy of malware detection.

There are 26 and 676 features of .asm uni grams and bi grams respectively. Similarly for .byte files, there are 257 and 66049 features for unigrams and bigrams respectively. The pixel array is 1D - flattened array of 800 pixel values.

### Models and Results

| Features Used | Model | Train loss | Test loss|   Accuracy|
| --- | --- | --- | --- | --- |
| ASM Uni | Random Forest |1.93 |1.89| 73.23 |
| Bytes Bi | Random Forest | 0.019 | 0.073 |  98.35 |
| ASM Uni + Bytes Bi | XGBoost |0.010 |0.037| 99.35 |
| ASM Uni + Bytes Bi | KNN |0.040 |0.069| 98.44 |
| ASM Uni+ Bytes Uni + ASM Bi | XGBoost  | 0.008 |0.017|99.72|
| ASM Bi + ASM Uni + Bytes Bi + ASM Pixel | Random forest  | 0.013 |0.011|100|

### Conclusion
One other conclusion that can be drawn from the table is that the model consisting of ASM Bi, ASM Uni, Bytes Bi, and ASM Pixel features achieved a perfect accuracy of 100% on both training and testing data when using the Random Forest model. This suggests that the combination of these features may be particularly effective in distinguishing between malware and benign files. However, it is important to note that the malware detection model might go out of date if it is not regularly trained on latest data.

# Stripper Well Problem

[Provide a brief description of the project here]

...

## Master's Dissertation

[Provide a brief description of the project here]