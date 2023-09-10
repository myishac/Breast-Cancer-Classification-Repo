[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10438460&assignment_repo_type=AssignmentRepo)
Project Instructions
==============================

# Breast Cancer Classification

**Introduction:** 

Mammogram results detect the presence of breast cancer, but according to the American Cancer Society, mammogram screenings miss approximately 1/8 breast cancers, reducing early treatment opportunities. Current diagnoses rely on medical expertise in interpreting mammogram images and are susceptible to human error. Machine learning methods can be used to improve this process to provide more reliable diagnoses.

**Objective:**

Incorrect classification of mammogram results can result in false negatives (detecting no cancer when present) which lead to untreated cases, or false positives (detecting cancer when not present), resulting in unnecessary anxiety and procedures. The aim of this project is to determine which machine learning methods are most effective in reducing these misclassifications, thus reducing false negatives and false positives.

**Data:**

The data used is obtained from the University of California, Irvine (UCI) machine learning repository. It consists of mammogram mass data attributes and target diagnosis collected by the Institute of Radiology at the University Erlangen-Nuremberg.

**Methodology:**

Previous analysis used Logistic Regression, Decision Trees and K-nearest neighbours to predict the target diagnosis, with each providing an improvement in classification over the current approach. This project will extend the analysis by exploring:
1. Feature engineering techniques such as Principal Component Analysis (PCA) to extract important combinations of features while preserving data trends, and reducing processing time;
2. Random Forests which combine multiple decision trees trained on random features to reduce variance and overfitting;
3. Support Vector Machines (SVM) which use kernels to map data to high dimensional feature space, and produce a decision boundary/hyperplane to create separation between classes (McGregor), and improve classification;
4. Artificial Neural Networks (ANN) which recognize patterns in the data, using 3 layers (input, hidden, output) linked by nodes to form networks. This deep learning method can improve as it learns from the data but may be less explainable than other methods.

**Works Cited**

Guide, Step. “Understanding Random Forest. How the Algorithm Works and Why it Is… | by Tony Yiu.” Towards Data Science, 12 June 2019, https://towardsdatascience.com/understanding-random-forest-58381e0602d2. Accessed 19 March 2023.

“Limitations of Mammograms | How Accurate Are Mammograms?” American Cancer Society, 14 January 2022, https://www.cancer.org/cancer/breast-cancer/screening-tests-and-early-detection/mammograms/limitations-of-mammograms.html. Accessed 19 March 2023.

“Mammographic Mass Data Set.” UCI Machine Learning Repository, 29 October 2007, https://archive.ics.uci.edu/ml/datasets/mammographic+mass. Accessed 19 March 2023.

McGregor, Milecia. “SVM Machine Learning Tutorial – What is the Support Vector Machine Algorithm, Explained with Code Examples.” freeCodeCamp, 1 July 2020, https://www.freecodecamp.org/news/svm-machine-learning-tutorial-what-is-the-support-vector-machine-algorithm-explained-with-code-examples/. Accessed 19 March 2023.

Vogel, Daniel. “Artificial Neural Network for Machine Learning — Structure & Layers.” Medium, https://medium.com/javarevisited/artificial-neural-network-for-machine-learning-structure-layers-a031fcb279d7. Accessed 19 March 2023.

This repo contains the instructions for a machine learning project.

**Execution:** 

To run the project, download following files from src:

    1. src > data > pre_processing.py
    
    2. src > features > build_features.py
    
    3. src > models > ann.py, decision_tree.py,knn.py, logistic_classifier.py, 
                      model_initializing.py, predict_model.py, random_forest.py, roc_curve.py, svm.py
    
    4. src > visualization > visualize.py
    
    5. data > raw > mammographic_masses.data
    
    Run project using main.py file.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── main.py    <- Makes src a Python module
       │
       ├── preprocessing data           <- Scripts to download or generate data and pre-process the data
       │   └── pre-processing.py
       │
       ├── features       <- Scripts to turn raw data into features for modeling
       │   └── build_features.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── model_initializing.py
       │   └── logistic_classifier.py
       │   └── decision_tree.py
       │   └── knn.py
       │   └── random_forest.py
       │   └── svm.py
       │   └── ann.py
       │   └── roc_curve.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py           

