import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

plt.style.use('seaborn-white')
RANDOM_STATE = 42

class Decision_Tree_Classifier:

    def __init__(self, X_train, y_train, max_depth=1, RANDOM_STATE=42):
        self.X_train = X_train
        self.y_train = y_train
        
        self.model = DecisionTreeClassifier(max_depth=max_depth,random_state=RANDOM_STATE)
     
    def get_model(self):
        return self.model
    
    def __str__(self):
        return "Decision Tree"  
    
    def fit(self, tune_fit="yes"):
       
        if tune_fit == "yes":
            self.model.fit(self.X_train, self.y_train)
            param_grid = {
                'min_samples_split': list(range(2, 12)),
                'min_samples_leaf': [10, 15],
                'max_depth': [2],
                'criterion': ['gini','entropy']
                }
            grid_search  = GridSearchCV(self.model, param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            self.model = grid_search.best_estimator_
            print(self.best_params)
            
        else:
            self.model = self.model.fit(self.X_train, self.y_train)
            print(self.model.get_params())
            
        return 
    
    def get_fit_model(self):
        final_fit = self.model.fit(self.X_train, self.y_train)
        return final_fit
    
    def predict(self):
        self.y_pred = self.model.predict(self.X_train)
        return self.y_pred
   
    def calculate_metrics_classification(self):
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_train, y_pred)
        precision = precision_score(self.y_train, y_pred, average='macro')
        recall = recall_score(self.y_train, y_pred, average='macro')
        f1 = f1_score(self.y_train, y_pred, average='macro')
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
        return accuracy, precision, recall, f1
       
    def plot(self):
        plt.figure(figsize=(15,10))
        tree.plot_tree(self.model, filled=True)
        plt.show()
        return
   
    def plot_importance(self):
        Importance = pd.DataFrame({'Importance':self.model.feature_importances_*100}, index=self.X_train.columns)
        Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
        plt.xlabel('Variable Importance')
        plt.gca().legend_ = None
        return
       
    def score(self):

        y_pred = self.predict()
        accuracy = accuracy_score(self.y_train,y_pred)
        report = classification_report(self.y_train, y_pred)
        return accuracy, report