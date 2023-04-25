# Data Science Portfolio
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
> For in depth details about this project, check out my blog here https://dev.to/vijethrai/solving-the-stripper-well-problem-15m9

### About Stripper Wells
A Stripper well is a low yield oil well. These wells have low operational costs and have tax breaks which makes it attractive from a business point of view. Almost 80% of the oil wells in the US are Stripper wells.

### Business Problem
The well has a lot of mechanical components and the breakdown is quite often compared to other wells. The breakdown occurs at surface level or down-hole level. In a business perspective, time is more valuable than small overhead costs. It is very inefficient to send the repair team without knowing where the failure has occured. Therefore it becomes a neccessity to create an algorithm which predicts where the failure has occured.

### Project Goal
The goal of the project is to predict whether the mechanical component has failed in the surface level or down-hole level. This information can be used to send the repair team to address the failure in either of the levels and save valuable time.

### Dataset
The dataset is provided by ConocoPhilips https://www.kaggle.com/competitions/equipfailstest/data. The dataset has 345 features which are taken from the sensors that collect a variety of information at surface and bottom levels. There are 50084 instances.

### EDA Conclusions
- The dataset is heavily imbalanced
- 172 features with greater than 25% missing values
- 6 features with outliers
- Many highly correlated features

### Preprocess
#### 1. Checking whether missing values are saying something

Maybe the missing values actually say something valuable regarding the equipment failure. I will check if this is the case by using Hamming distance between features with missing values and the target class. The logic is, if there is any relation with failure and nan values, the hamming distance with be very high. I am removing all features that do not have a significant value of hamming distance in this preprocess stage.

#### 2. Removing highly correlated features

Removing all features that have corelation greater than 0.85. This will lead to reduction in data dimension and also boosting algorithms will work better.

#### 3. Scaling feature

Normalizing all the features so that it can be readily used for any model without extra scaling techniques.

### Models

| Model | F1 Score | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- |
| Random Forest | 0.89 | 0.90 | 0.70 | 99.995 |
| XGBoost | 0.92 | 0.84 | 0.84 | 99.996 |
| AdaBoost | 0.89 | 0.92 | 0.68 | 99.995 |
| Decision Tree | 0.80 | 0.58 | 0.66 | 99.989 |
| ANN | 0.78 | 0.58 | 0.85 | 99.990 |
| Best Ensemble | 0.82 | 0.54 | 0.84 | 99.989 |

### Conclusion
New features impact the accuracy of the model by around 30%. XGBoost has the best F1 score. 

## Master's Dissertation

[Provide a brief description of the project here]