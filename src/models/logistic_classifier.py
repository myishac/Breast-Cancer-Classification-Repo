from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

#Used for training only
class Logistic_Classifier:
    def __init__(self, X_train, y_train, random_state=42):
        """Initializes the class with the input features X and target variable y,
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

        self.best_params = {}
        self.best_score = 0
        
        self.X_train = X_train
        self.y_train = y_train
               
        self.random_state = random_state
        self.model = LogisticRegression(fit_intercept=True, random_state=self.random_state)
    
        return 
        
    def get_model(self):
        return self.model
    
    def __str__(self):
        return "Logistic Regression"
    
    def fit(self, tune_fit="yes"):
        """Trains the model using the input data X and y.
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
            param_grid = {'C':(0.1, 1),
                          'solver' :['newton-cg','lbfgs', 'liblinear'],
                          'penalty':['l1','l2']
                         }
                                       
            grid_search = GridSearchCV(self.model, param_grid, 
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
        """Predicts the target variable of the test data using the trained logistic regression model.
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
        accuracy = accuracy_score(self.y_train, y_pred)
        report = classification_report(self.y_train, y_pred)
        
        return accuracy, report