# to catch any warnings
import warnings
warnings.filterwarnings("ignore")

# Load libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import packages that will be used in PCA
DIVIDE = '-' * 60


class Visualize:
    def __init__(self, mammoData):
        self.mammoData = mammoData
        self.finalData = None
        self.pcaData_toSplit = None
        
    def eda(self):
        # We now look at the distribution of variables in the data and how the attributes are related to the target. 
        # Based on the following charts, the dataset seems well-balanced with respect to the target (51% benign and 49% malignant).
       
        #Look at distribution of target variable in the data
        valueSeverity = self.mammoData['Severity'].value_counts(normalize = True)
        print("Value Severity:")
        print(valueSeverity)
        # Data is balanced with about 51% benign cases and 49% malignant cases
        
        #Look at distribution of target variable in the data
        print("Look at distribution of target variable in the data")
        plotData = self.mammoData['Severity'].value_counts(normalize=True)*100
        newData = plotData.to_frame().reset_index()
        newData.columns = ['Severity', 'Percentage']
        # newData.head()
        
        sns.barplot(data=newData, x='Severity', y ='Percentage');
        plt.title('Percentage of Severity by Outcome', fontsize = 16);
        plt.show()
    
        # Look at Distribution of Density variable
        print("Look at Distribution of Density variable")
        __, axes = plt.subplots(1, 1, figsize=(10, 8))
        # Plot frequency plot/ histogram to look for distribution as well as outliers
        sns.histplot(x="Density", kde=False, data=self.mammoData.sort_values('Density'), ax=axes, bins=4);
        axes.set(xlabel="Density");
        axes.xaxis.label.set_size(18)
        axes.yaxis.label.set_size(18)
        axes.tick_params('y', labelsize = 14);
        axes.tick_params('x', labelsize = 14);
        plt.xticks([0,1,2,3,4]);
        plt.title('Distribution of Density', fontsize=16);
        plt.show()

        self.mammoData['Margin'].value_counts(ascending =False)
        
        __, axes = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot frequency plot/ histogram to look for distribution as well as outliers
        print("Plot frequency plot to look for distribution as well as outliers")
        sns.histplot(x="Margin", kde=True, data=self.mammoData.sort_values('Margin'), ax=axes, bins=5);
        axes.set(xlabel="Margin");
        axes.xaxis.label.set_size(18)
        axes.yaxis.label.set_size(18)
        axes.tick_params('y', labelsize = 14);
        axes.tick_params('x', labelsize = 14);
        plt.title('Distribution of Margin', fontsize=16);
        plt.show()
        
        # Most of the mammogram masses are in either the 1 or 4 category i.e. circumscribed or ill-defined
        
        self.mammoData['Shape'].value_counts(ascending =False)
        
        __, axes = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot frequency plot/ histogram to look for distribution as well as outliers
        sns.histplot(x="Shape", kde=True, data=self.mammoData.sort_values('Shape'),ax=axes, bins=4);
        axes.set(xlabel="Shape");
        axes.xaxis.label.set_size(18)
        axes.yaxis.label.set_size(18)
        axes.tick_params('y', labelsize = 14);
        axes.tick_params('x', labelsize = 14);
        plt.title('Distribution of Shape', fontsize=16);
        plt.show()
        
        # Category 3 (lobular) contains the least number of records
        
        # Look at distribution of Age
        print("Look at distribution of Age")
        sns.displot(data = self.mammoData['Age'], kde=True, bins = 40)
        plt.title('Distribution of Age', fontsize=16)
        axes.set(xlabel="Age")
        axes.xaxis.label.set_size(18)
        axes.yaxis.label.set_size(18)
        plt.show()

        print(DIVIDE)


    def pred_respon_relation(self):

        # We now examine how each of the feature variables is related to the target. 
        # Based on these charts we note the following:
            # a. Density category 2 (iso) has a lower proportion of breast cancer than the other categories.
            # b. Shape categories 1 and 2 have lowest proportion of breast cancer. 
                # The incidence rate is about 50% for shape category 3, and increases to almost 80% for Shape category 4.
            # c. Margin category 1 has low proportion of breast cancer (about 12%). 
                # The incidence rate increases with each subsequent category, and is highest for Margin category 5.
            # d. Breast cancer appears to be more likely at higher ages.

        #create a stacked bar chart to show incidence of breast cancer by each Density class
        pd.crosstab(self.mammoData['Density'],self.mammoData['Severity'],normalize='index').plot.bar(stacked=True)
        plt.title("Proportion of Breast Cancer Diagnosis by Density", fontsize=16);
        #plt.legend(loc='center right', title='Severity')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Severity');
        plt.show()

        #Desnity category 2 (iso) has a lower incidence of breast cancer than the other categories.

        #create a stacked bar chart to show incidence of breast cancer by each shape class
        pd.crosstab(self.mammoData['Shape'],self.mammoData['Severity'],normalize='index').plot.bar(stacked=True)
        plt.title("Proportion of Breast Cancer Diagnosis by Shape", fontsize=16);
        #plt.legend(loc='center right', title='Severity')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Severity');
        plt.show()

        #Shape categories 1 and 2 have lowest incidence of breast cancer. The incidence rate is about 50% for shape category 3,
        #and increases to almoat 80% for Shape category 4.

        #create a stacked bar chart to show incidence of breast cancer by each margin class
        pd.crosstab(self.mammoData['Margin'],self.mammoData['Severity'],normalize='index').plot.bar(stacked=True)
        plt.title("Proportion of Breast Cancer Diagnosis by Margin", fontsize=16);
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Severity');
        plt.show()
        #Margin category 1 has low incidence of breast cancer. The incidence rate increases with each subsequent category,
        #and is highest for Margin category 5.

        sns.boxplot(data=self.mammoData, x="Severity", y="Age")
        plt.title('Distribution of Age by Breast Cancer Diagnosis', fontsize=16)
        sns.set(rc = {'figure.figsize':(8,6)});
        plt.show()
        print(DIVIDE)
        #Incidence of breast cancer appears to be more likely at higher ages