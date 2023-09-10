# to catch any warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from IPython.display import display

# Load libraries
from pre_processing import Pre_processing
from visualize import Visualize
from model_initializing import Model_Initializing
from build_features import Build_Features
from logistic_classifier import Logistic_Classifier
from decision_tree import Decision_Tree_Classifier
from knn import KNN_Classifier
from random_forest import Random_Forest_Classifier
from svm import SVM_Classifier
from ann import Neural_Network_Classifier
from predict_model import Predict_Model
from roc_curve import Roc_Curve

DIVIDE = '-' * 60

print("LOAD AND DESCRIBE DATA \n")
#LOAD AND DESCRIBE DATA
initialData = Pre_processing("mammographic_masses.data")
initialData.read_data()
initialData.describe_data()

print("REVIEW AND CLEAN DATA \n")
#REVIEW AND CLEAN DATA
cleaned_data = initialData.review_clean()

print("MODEL PLANNING - EXPLORARTORY DATA ANALYSIS \n")
#MODEL PLANNING - EXPLORARTORY DATA ANALYSIS
model = Visualize(cleaned_data)
#model.define_data()
model.eda()

print("RELATIONSHIPS BETWEEN PREDICTORS AND RESPONSE \n")
#RELATIONSHIPS BETWEEN PREDICTORS AND RESPONSE
model.pred_respon_relation()

print("FEATURE ENGINEERING \n")
#FEATURE ENGINEERING
features = Build_Features(cleaned_data)

print("PRINCIPAL COMPONENT ANALYSIS\n")
#PCA
dataPCA = features.pca()

print("ONE-HOT ENCODING \n")
# ONE-HOT-ENCODING
finalData = features.feature_engineering()

print("SPLITTING DATA INTO TRAIN AND TEST \n")
# SPLITTING DATA INTO TRAIN AND TEST
model = Model_Initializing(finalData)
X_train_s, y_train, X_test_s, y_test = model.split_data()

print("CREATING MODELS \n")
# CREATING MODELS 

print(DIVIDE)

# LOGISTIC REGRESSION
print("MODEL 1: LOGISTIC REGRESSION \n")

#training 
print("Testing on train data: \n")
# Create an instance of the Logistic classifier
lr_model = Logistic_Classifier(X_train_s, y_train)

# Fit the Logistic model to the training data
lr_model.fit(tune_fit="yes")

# Make predictions on the train data
pred_lr = lr_model.predict()

accuracy_lr, classification_report_lr = lr_model.score()
print("The training accuracy achieved by tuned lr model:", accuracy_lr)
print('#'*60)
print("The training classification report of tuned lr model: \n", classification_report_lr)


print(DIVIDE)


# DECISION TREE
print("MODEL 2: DECISION TREE \n")

# training
print("Testing on train data: \n")
# Create an instance of the decision tree classifier with 
dt_model = Decision_Tree_Classifier(X_train_s, y_train)

# Fit the model to the decision tree training data
dt_model.fit(tune_fit="yes")

# Make predictions on the train data
pred_dt = dt_model.predict()

accuracy_dt, classification_report_dt = dt_model.score()
print("The training accuracy achieved by tuned dt model:", accuracy_dt)
print('#'*60)
print("The training classification report of tuned dt model: \n", classification_report_dt)
dt_model.plot()


print(DIVIDE)


# K-NEAREST NEIGHBOURS
print("MODEL 3: K-NEAREST NEIGHBOURS \n")

#training
print("Testing on train data: \n")
# Create an instance of the KNN classifier with default hyperparameters
knn_model = KNN_Classifier(X_train_s, y_train)

# Fit the KNN model to the training data
knn_model.fit(tune_fit="yes")

# Make predictions on the train data
pred_knn = knn_model.predict()

accuracy_knn, classification_report_knn = knn_model.score()
print("The training accuracy achieved by tuned knn model:", accuracy_knn)
print('#'*60)
print("The training classification report of tuned knn model: \n", classification_report_knn)


print(DIVIDE)


# RANDOM FOREST
print("MODEL 4: RANDOM FOREST \n")

#training
print("Testing on train data: \n")
# Create an instance of the Random Forest classifier with default hyperparameters
rf_model = Random_Forest_Classifier(X_train_s, y_train)

# Fit the Random Forest model to the training data
rf_model.fit(tune_fit="yes")

# Make predictions on the train data
pred_rf = rf_model.predict()

accuracy_rf, classification_report_rf = rf_model.score()
print("The training accuracy achieved by tuned rf model:", accuracy_rf)
print('#'*60)
print("The training classification report of tuned rf model: \n", classification_report_rf)

rf_model.plot_feature_importance(3)


print(DIVIDE)


# SVM
print("MODEL 5: SUPPORT VECTOR GRAPH \n")

#training
print("Testing on train data: \n")
# Create an instance of the SVM classifier with default hyperparameters
svm_model = SVM_Classifier(X_train_s, y_train)

# Fit and tune the SVM model to the training data
svm_model.fit(tune_fit="yes")

# Make predictions on the train data
pred_svm = svm_model.predict()

accuracy_svm, classification_report_svm = svm_model.score()
print("The training accuracy achieved by tuned svm model:", accuracy_svm)
print('#'*60)
print("The training classification report of tuned svm model: \n", classification_report_svm)


print(DIVIDE)


# ANN
print("MODEL 6: ARTIFICIAL NEURAL NETWORK \n")

#training
print("Testing on train data: \n")
# Create an instance of the ANN classifier with default hyperparameters
ann_model = Neural_Network_Classifier(X_train_s,y_train)

# Fit and tune the ANN model to the training data
ann_model.fit(tune_fit="yes")

# Make predictions on the train data
pred_ann = ann_model.predict()

accuracy_ann, classification_report_ann = ann_model.score()
print("The training accuracy achieved by tuned ann model:", accuracy_ann)
print('#'*60)
print("The training classification report of tuned ann model: \n", classification_report_ann)


print(DIVIDE)


#CURRENT METHODOLOGY FOR CLASSIFYING MAMMOGRAM MASSES
print("CURRENT METHODOLOGY FOR CLASSIFYING MAMMOGRAM MASSES \n")
initialData.confusion_matrix()
print(DIVIDE)


# Testing all models with test data
print("Testing Logistic Regression on test data: \n")
pred_lr = Predict_Model(lr_model,X_test_s,y_test)
pred_lr.confMatrix()
lr_acc, lr_prec, lr_far, lr_fnr = pred_lr.calcMetrics()
print(DIVIDE)

print("Testing Decision Tree on test data: \n")
pred_dt = Predict_Model(dt_model,X_test_s,y_test)
pred_dt.confMatrix()
dt_acc, dt_prec, dt_far, dt_fnr = pred_dt.calcMetrics()
print(DIVIDE)

print("Testing KNN on test data: \n")
pred_knn = Predict_Model(knn_model,X_test_s,y_test)
pred_knn.confMatrix()
knn_acc, knn_prec, knn_far, knn_fnr = pred_knn.calcMetrics()
print(DIVIDE)

print("Testing Random Forest on test data: \n")
pred_rf = Predict_Model(rf_model,X_test_s,y_test)
pred_rf.confMatrix()
rf_acc, rf_prec, rf_far, rf_fnr = pred_rf.calcMetrics()
print(DIVIDE)

print("Testing SVM on test data: \n")
pred_svm = Predict_Model(svm_model,X_test_s,y_test)
pred_svm.confMatrix()
svm_acc, svm_prec, svm_far, svm_fnr = pred_svm.calcMetrics()
print(DIVIDE)

print("Testing ANN on test data: \n")
pred_ann = Predict_Model(ann_model,X_test_s,y_test)
pred_ann.confMatrix()
ann_acc, ann_prec, ann_far, ann_fnr = pred_ann.calcMetrics()
print(DIVIDE)

# ROC Curve
list_models = [lr_model, dt_model, knn_model, rf_model, svm_model, ann_model]
roc = Roc_Curve(list_models, X_test_s, y_test)
roc.plot_roc_curve()

# Comparing all models
models_1 = [('Logistic Regression', lr_acc, lr_prec, lr_far, lr_fnr),
         ('Decision Tree', dt_acc, dt_prec, dt_far, dt_fnr),
         ('K-Nearest Neighbours', knn_acc, knn_prec, knn_far, knn_fnr),
         ('Random Forest', rf_acc, rf_prec, rf_far, rf_fnr),
         ('Support Vector Machine', svm_acc, svm_prec, svm_far, svm_fnr),
         ('Artificial Neural Network', ann_acc, ann_prec, ann_far, ann_fnr)]

t1 = pd.DataFrame(data=models_1,
                  columns = ['Model', 'Accuracy','Precision',
                             'False Alarm Rate', 'False Negative Rate']).set_index('Model')

models_2 = [('Logistic Regression', accuracy_lr*100, lr_acc),
         ('Decision Tree', accuracy_dt*100, dt_acc),
         ('K-Nearest Neighbours', accuracy_knn*100, knn_acc),
         ('Random Forest', accuracy_rf*100, rf_acc),
         ('Support Vector Machine', accuracy_svm*100, svm_acc),
         ('Artificial Neural Network', accuracy_ann*100, ann_acc)]

t2 = pd.DataFrame(data=models_2,
                  columns = ['Model', 'Training Accuracy',
                             'Test Accuracy']).set_index('Model')

display(t1.round(0))
display(t2.round(1))

# Discussion
# Comparing the classificaiton models explored in this analysis, 
# it appears that the K-Nearest Neighbours produced the highest test accuracy, 
# while the Decision Tree has the lowest test accuracy. 
# The remaining models have similar test accuracies around 80% . 
# The KNN model also produced a false alarm rate of 28% and a false negative rate of 10%, which were among the lowest of all the models.
# Plotting the sensitivity measure against the false positive rate produces the Receiver Operating Characteristic (ROC) curve. 
# This curve can be used to determine the optimal threshold at which the classes should be defined. 
# The Area Under the Curve (AUC) provides the total measure of performance across all possible thresholds for a model. 
# Comparing the AUC measure, the logistic regression model had the highest value of 87%, 
# while the AUC for the neural network was just slightly lower at 86%. Again, the decision tree performed the worst based on AUC.