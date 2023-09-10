# Imports
# to catch any warnings
import warnings
warnings.filterwarnings("ignore")

# Load libraries
import numpy as np
import pandas as pd

DIVIDE = '-' * 60

class Pre_processing:
    # create a dataframe with only the BI-RADS assessment and Severity columns
    # Remove any rows with missing bi-rads values
    
    def __init__(self, dataname):
        
        self.dataname = dataname
        self.mammogramdata = None 
        self.mammoData = None     

    def read_data(self):
        temp = pd.read_csv(self.dataname, names = ["BI-RADS assessment", "Age", "Shape", "Margin", "Density", "Severity"])
        self.mammogramdata = temp.replace("?", np.nan)
        
        
    def describe_data(self):
        # self.mammogramdata = data
        # newdata = self.mammogramdata.replace("?", np.nan, inplace=True)
        # data = pd.read_csv(self.mammogramdata, names = ["BI-RADS assessment", "Age", "Shape", "Margin", "Density", "Severity"])
        # newdata = self.mammogramdata.replace("?", np.nan)
        sample = self.mammogramdata.head(5)
        print(sample)
        print(DIVIDE)

        # Look at the makeup of the data
        print("Original mammoDate Info:")
        print()
        self.mammogramdata.info()
        #961 rows of 5 variables and 1 target (severity)
        print(DIVIDE)
    
    def confusion_matrix(self):
        data = self.mammogramdata
        
        birads = data.loc[:, ['BI-RADS assessment', 'Severity']]


        birads['BI-RADS assessment'] = birads['BI-RADS assessment'].replace({'0':np.nan,'6':np.nan, '55':'5'}).dropna()
        b_val_count = birads.value_counts()
        print("BI-RADS Value Count:")
        print()
        print(b_val_count)
        print(DIVIDE)

        # replace BI-RADS assessment with a 0 or 1, based on information from *****
        birads['BI-RADS assessment'] = birads['BI-RADS assessment'].replace({'1':0, '2':0, '3':0, '4':1, '5':1})
        birads['BI-RADS assessment'].value_counts()

        #calculate values for confusion matrix
        truepositive = sum((birads['BI-RADS assessment'] == 1) & (birads['Severity'] == 1))
        falsepositive = sum((birads['BI-RADS assessment'] == 1) & (birads['Severity'] == 0))
        falsenegative = sum((birads['BI-RADS assessment'] == 0) & (birads['Severity'] == 1))
        truenegative = sum((birads['BI-RADS assessment']  == 0) & (birads['Severity'] == 0))
        print("TP, FP, FN, TN")
        print(truepositive, falsepositive, falsenegative, truenegative)
        print(DIVIDE)

        # Create and display confusion matrix
        birad_cm = np.array([[truenegative, falsepositive],[falsenegative, truepositive]])
        print("BI_RADS Confusion Matrix:")
        print()
        print(birad_cm)
        print(DIVIDE)

        br_acc = round((truepositive+truenegative)/(len(birads))*100,2)
        br_prec = round(truepositive/(truepositive+falsepositive)*100,2)
        br_far = round(falsepositive/(falsepositive+truenegative)*100,2)
        br_fnr = round(falsenegative/(falsenegative+truepositive)*100,2)

        print("The accuracy for BI-RADS assessment is:",br_acc,'%')
        print("The precision for BI-RAD S assessment is:",br_prec,'%')
        print("The false alarm rate for BI-RADS assessment is:",br_far,'%')
        print("The false negative Rate for BI-RADS assessment is:",br_fnr,'%')

    def review_clean(self):
        # From this summary it appears that the majority of mammogram masses (over 90%) are classified as malignant under
        # the BI-RADS assessment. 

        #The BI-RADS variable is not-predictive so it is removed from the data
        self.mammoData = self.mammogramdata.drop(['BI-RADS assessment'], axis=1) # should make a text file
        sample = self.mammoData.head()
        print("mammoData Head:")
        print()
        print(sample)
        print(DIVIDE)

        #Check for missing values
        missing_mammvalues = self.mammoData.isnull().sum().sort_values(ascending = False)
        print("Missing mammoData Values:")
        print()
        print(missing_mammvalues )
        print(DIVIDE)
        

        # missing values exist for each one of the predictors. No missing values in target.


        #Most of the records with missing values may have to be dropped
        #Can examine missing Age which occurs in only 5 cases

        #change Age to integer
        # substitute median for the missing Age values
        self.mammoData['Age'] = self.mammoData['Age'].fillna(self.mammoData['Age'].median()).astype('int')
        sample2 = self.mammoData['Age'].describe()
        print("Age Description:")
        print()
        print(sample2)
        print(DIVIDE)


        #Drop the records that still contain missing values
        self.mammoData = self.mammoData.dropna()
        print("mammoData Info:")
        print()
        self.mammoData.info()
        
        #there are 836 records remaining
        print(DIVIDE)

        # Change Density variable to integer as it is an ordinal value
        self.mammoData['Density'].astype('int')
        sample4 = self.mammoData['Density'].value_counts(ascending=False)
        print("Density Value Counts:")
        print()
        print(sample4)

        self.mammoData.to_csv('mammoData.txt', index = False)
        
        print(DIVIDE)


        
        return self.mammoData
