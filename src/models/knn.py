import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#Used for training only
class KNN_Classifier:
    def __init__(self, X_train, y_train,  random_state=42):
        """
        Initialize the KNN classifier.

        Parameters:
        -----------
            X (pandas.DataFrame): The feature matrix of shape (n_samples, n_features).
            y (pandas.Series): The target vector of shape (n_samples,).
            n_neighbors (int, optional): The number of nearest neighbors to use in classification. Defaults to 5.
            test_size (float, optional): The proportion of samples to use for testing. Defaults to 0.2.
            random_state (int, optional): The random state to use for splitting the data. Defaults to 42.
        """
        
        self.model = KNeighborsClassifier()
        self.best_params = {}
        self.best_score = 0
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        
        return
        
    def get_model(self):
        return self.model
    
    def __str__(self):
        return "KNN"
    
    def fit(self, tune_fit="yes"):
        """
        Fit the KNN model to the training data.

        Parameters:
        -----------
            tune_fit (str, optional): Whether to perform hyperparameter tuning. If "yes", performs a grid search to find
                the best hyperparameters. Defaults to "no".

        Raises:
        -----------
            ValueError: If `tune_fit` is not "yes" or "no".

        Returns:
        -----------
            None
        """
        if tune_fit=="yes": 
            param_grid = {'n_neighbors' : np.arange(3,11,2),
               'weights' : ['distance', 'uniform'],
               'metric' : ['minkowski', 'manhattan']
            }
            grid_search = GridSearchCV(self.model, param_grid, 
                                       #scoring = 'accuracy',
                                       cv=5)
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
        """
        Predict the target values for the test data.

        Parameters:
        -----------
        None

        Returns:
        -----------
            numpy.ndarray: The predicted target values of shape (n_samples,)
        """
        return  self.model.predict(self.X_train)
    
    def score(self):
        """
        Calculate the accuracy score and classification report for the KNN model.

        Parameters:
        -----------
        None

        Returns:
        -----------
        accuracy score: The ratio of the correctly predicted observations to the total observations.
        
        classification report: A text report of the main classification metrics such as precision, recall, f1-score and support for each class.
        """

        y_pred = self.predict()
        accuracy = accuracy_score(self.y_train, y_pred)
        report = classification_report(self.y_train, y_pred)
        
        return accuracy, report

