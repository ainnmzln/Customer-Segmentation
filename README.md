![badge](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

# Customer-Segmentation

# Summary 

Previously, the sales team has classified all customers into 4 segments (A, B, C, D ). From the previous data, they plan to use the same strategy on new markets and have identified new potential customers.

# Datasets
This projects is trained with  [Customer Segmentation](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) dataset.

# Requirements 

This project is created using Spyder as the main IDE. The main frameworks used in this project are Pandas, Numpy, Sklearn, TensorFlow and Tensorboard. 

# Methodology

This project contains three .py files. The module.py is for module that will be used in this project The train.py and deploy.py are for training and to depploy and to test the new_customers dataset. The flow of the projects are as follows:

## 1. Importing the libraries and dataset

The data are loaded from the dataset and usefull libraries are imported.

## 2. Exploratory data analysis

### 2.1 Data visualization

The labels and target are plotted to see the relation in this discrete dataset.

### Ever_Married vs Segmentation

![Ever_Married vs Segmentation](https://github.com/ainnmzln/Customer-Segmentation/blob/main/images/Ever%20married%20vs%20segmentation.png)

From the graph, it is show that unmarried customers most likely to fall to Category D and married customers contribute to Category C.

### Gender vs Segmentation

![Gender vs Segmentation](https://github.com/ainnmzln/Customer-Segmentation/blob/main/images/Gender%20vs%20segmentation.png)

From the graph, it is show that both male and female customers most likely to fall to Category D.

### Profession vs Segmentation

![Profession vs Segmentation](https://github.com/ainnmzln/Customer-Segmentation/blob/main/images/Profession%20vs%20Segmentation.png)

From the graph, it is show that customers who works in Healthcare contribute to Category D and artist fall to Category C. 

### 2.2 Data cleaning 

The datasets is cleaned with necessary step. Since there are a lots of missing value in some coulmns, the data need to be cleaned with Simple Imputer. Unnecessary column is dropped such as ID and Working_Experince.

### 2.3 Data pre-processing

The data is scaled with Min Max Scaler. The dataset is then split to test and train set with ratio 7:3.


## 3. Deep Learning model

A LSTM modelled is developed to train the data. However, the model showed no training progress since the data shows not much correlation between features and labels.

## 4. Deep Learning model

Machine learning model and pipeline are created to train the muticclass classification data such as:

1. Logistic regression
2. K Neighbors Classifier
3. Random Forest Classifier
4. Support Vector Classifier
5. Decision Tree Classifier

## 5. Model Prediction and Accuracy

The results with the best accuracy score is SVC Classifier with 50 % accuracy score.

![](https://github.com/ainnmzln/Customer-Segmentation/blob/main/images/SCORE.png)

## 6. Deployment
The data is then deployed to predict with the new_customers dataset.

