# Run this cell to suppress all FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#for graphs
import matplotlib.pyplot as plt

#models to run
from sklearn import tree

#metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42

#Used for training only
class Random_Forest_Classifier:
    def __init__(self, X_train, y_train, max_depth=1, random_state=42):
        """Initializes the RandomForest class with the input features X and target variable y,
        and the random state used for reproducibility.
        Parameters:
        -----------
        X : pandas.DataFrame
            The input feature matrix of shape (n_samples, n_features).
        y : pandas.Series
            The target variable of shape (n_samples,).
        random_state : int, default=RANDOM_STATE, which is 42
            The seed value for random number generator used to split the data.
        Returns:
        --------
        None"""

        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
            
        self.model = RandomForestClassifier(max_depth=1,random_state=self.random_state)
        self.best_params = {} 
        self.best_score = 0 
        return
     
    def get_model(self):
        return self.model
    
    def __str__(self):
        return "Random Forest"
    
    def fit(self, tune_fit="yes"):
        """Trains the random forest model using the input data X and y.
        If tune_fit is set to "yes", it tunes the hyperparameters using GridSearchCV(), otherwise, with default parameters.
        It then stores the best hyperparameters and best estimator in the attributes best_params and model, respectively.
        Parameters:
        -----------
        tune_fit : str, default="yes"
            If "yes", tune the hyperparameters using GridSearchCV(), otherwise use default parameters.
        Returns:
        --------
        None
        """

        if tune_fit=="yes":
            param_grid = {'n_estimators' : [2,5,12],
                          'max_depth' : range(1,5),
                          'criterion' :['gini', 'entropy']}
            
            grid_search = GridSearchCV(self.model, param_grid=param_grid, cv= 3)
            grid_search.fit(self.X_train, self.y_train)
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            self.model = grid_search.best_estimator_
            print(self.best_params)    
            
        elif tune_fit=="no":
            self.model = self.model.fit(self.X_train, self.y_train)
            print(self.model.get_params()) 
            
        else:
            raise ValueError("Invalid value for `tune_fit`. Must be either 'yes' or 'no'.")
        
        return 
    
    def get_fit_model(self):
        final_fit = self.model.fit(self.X_train, self.y_train)
        return final_fit
            
    def predict(self):
        """Predicts the target variable of the test data using the trained random forest model.
        Returns the predicted target variable values.
        Parameters:
        -----------
        None
        Returns:
        --------
        numpy.ndarray: The predicted target variable values
        
        """
        return self.model.predict(self.X_train)
    
    def score(self):
        """Computes the accuracy score and classification report for the predicted target variable and the actual test target variable.
        Parameters:
        -----------
        None
        Returns:
        --------
        accuracy score: The ratio of the correctly predicted observations to the total observations.
        
        classification report: A text report of the main classification metrics such as precision, recall, f1-score and support for each class."""
        
        y_pred = self.predict()
        #accuracy = self.best_score 
        accuracy = accuracy_score(self.y_train,y_pred)
        report = classification_report(self.y_train, y_pred)
        
        return accuracy,report 
    
    def plot_decision_tree(self):
        """Plots the first decision tree in the trained random forest model.
        Parameters:
        -----------
        None
        Returns:
        -------- 
        None
        """

        # Extract the first tree from the forest
        estimator = self.model.estimators_[0]

        # Plot the decision tree
        plt.figure(figsize=(12, 8)) 
        tree.plot_tree(estimator, feature_names=self.X_train.columns, filled=True)
        plt.draw() 
        
        return
    
    def plot_feature_importance(self, top=10):
        """Plots a horizontal bar chart of the top (by default 10) important features in the random forest model.
        Parameters:
        -----------
        top: The number of top important features to display. Default is 10.
        Returns: 
        -----------
        None"""
        importances = self.model.feature_importances_
        indices = importances.argsort()
        plt.figure(figsize = (10,10))
        plt.title("Feature Importance")
        plt.barh([self.X_train.columns[i] for i in indices[-top:]], [importances[i] for i in indices[-top:]])
        plt.yticks(range(top), [self.X_train.columns[i] for i in indices[-top:]])
        plt.xlabel("Relative Importance")
        plt.draw()
        plt.show()
        
        return