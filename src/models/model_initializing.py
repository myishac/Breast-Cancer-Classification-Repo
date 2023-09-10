
# to catch any warnings
import warnings
warnings.filterwarnings("ignore")

# Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from visualize import pca
DIVIDE = '-' * 60

class Model_Initializing:
    
    def __init__(self, finalData):
        self.finalData = finalData
        self.X_train = None
        self.X_test = None
        self.X_train_s = None
        self.y_train = None
        self.X_test_s = None
        self.y_test = None
        

    def split_data(self):
        #Split data into train and test
        # We start by splitting the data into a training set and a test set. 
        # The test set will be set aside to be used for comparing the performance of models only after all the models have been fitted. 
        # When building models, it is possible that the model is over-fitted to data used for training and will thus perform poorly on unseen data. 
        # This is usually indicated by high training data accuracy, but low validation data accuracy. 
        # To avoid overfitting, we can
            # a. further split the training data into separate train and validation datasets. 
            # The train dataset can be used for training the model, while the validation dataset is used to determine 
            # how well the model predicts on a new set of data
        # OR
            # b. use k-fold cross-validation to repeatedly split the training data into separate folds 
            # where each fold contains a set of train data as well as validation data. 
            # This is set up such that each record is in the validation set for at least one-fold.
                
        #Cross validation will be used to fit the models
        #the test set will be used to compare the performance of each of the model types
        trainData, testData = train_test_split(self.finalData, test_size=0.2, random_state=42)
        self.X_train = trainData.drop(['Severity'], axis=1)
        self.y_train = trainData.loc[:,'Severity']
        
        self.X_test = testData.drop(['Severity'], axis=1)
        self.y_test = testData.loc[:,'Severity']
        print("X_test:")
        print()
        print(self.X_test.head())
        print(DIVIDE)
        # Normalizing the features for both train and test
        
        # Scale the features
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train_s = scaler.transform(self.X_train)  
        self.X_test_s = scaler.transform(self.X_test)
        
        col_names = list(self.X_train.columns.values)
        self.X_train_s = pd.DataFrame(self.X_train_s, columns=col_names)
        print("X_train_s:")
        print()
        print(self.X_train_s.head())
        print(DIVIDE)
        
        self.X_test_s = pd.DataFrame(self.X_test_s, columns=col_names)
        print("X_test_s:")
        print()
        print(self.X_test_s.head())
        print(DIVIDE)
        
        return self.X_train_s, self.y_train, self.X_test_s, self.y_test
        
