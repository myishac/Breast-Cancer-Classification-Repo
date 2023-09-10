# to catch any warnings
import warnings
warnings.filterwarnings("ignore")

# Load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import packages that will be used in model building and evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DIVIDE = '-' * 60

class Build_Features:
    def __init__(self, mammoData):
        self.mammoData = mammoData
        # self.pcaData = pcaData
        self.X_train = None
        self.X_test = None
        self.X_train_s = None
        self.y_train = None
        self.X_test_s = None
        self.y_test = None

    def pca(self):
        #Explore Principle Components Analysis 
        # Principal components analysis (PCA) is an unsupervised learning method that can be used to identify key variables, 
        # or to identify the presence of outliers in the data. 
        # In the mammographic mass data, there are only 4 features, so it is not considered high-dimensional.
        # However, we will explore the application of PCA to better understand the relationships between the features.
        # Based on the PCA analysis, we noted the following:
            # a. For the first component, margin has the strongest effect
            # b. For the second and third components, age and density have a strong effect
            # c. For the fourth component shape has the most impact
        #Create data without target variable
        mammo_features = self.mammoData.drop(['Severity'], axis=1)
        mammo_target = self.mammoData['Severity']

        #Scale the feature columns
        scaler = StandardScaler()
        scaler.fit(mammo_features)
        mammoFeatures_s = scaler.transform(mammo_features)

        col_names = list(mammo_features.columns.values)
        mammoFeatures_s = pd.DataFrame(mammoFeatures_s, columns=col_names)
        print("Scaled Mammogram Data (mammoFeatures_s):")
        print()
        print(mammoFeatures_s.head())
        print(DIVIDE)

        #fit PCA transformation
        pca = PCA()
        pca_mammo = pca.fit_transform(mammoFeatures_s)

        # Reformat and view results
        loadings = (pd.DataFrame(pca.components_, columns=['pc1', 'pc2', 'pc3','pc4'],
                                    index=mammo_features.columns))
        print("Loadings:")
        print(loadings)
        print(DIVIDE)

        mammoData_pca = pd.DataFrame(pca_mammo, columns = ['pc1', 'pc2', 'pc3','pc4'])
        print("Mammogram PCA Data:")
        print()
        print(mammoData_pca.head())
        print(DIVIDE)

        #Based on the PCA analysis, we noted the following:
        # For the first component, age and margin have a strong effect
        # For the second and third components, age and density have a strong effect
        # For the fourth component shape has the most impact

        #Create a plot of the target based on the first 2 principal components
        mammo_target = mammo_target.reset_index(drop=True)
        self.pcaData_toSplit = pd.concat([mammoData_pca, mammo_target], axis=1)
        plt.figure(figsize=(6,6))
        sns.scatterplot(
            x="pc1", y="pc2",
            hue= "Severity",
            #palette=sns.color_palette("hls", 10),
            data=self.pcaData_toSplit,
            legend="full",
            alpha=0.3
        )
        plt.show()
        print("PCA Data to Split:")
        print()
        print(self.pcaData_toSplit.head())
        print(DIVIDE)
        
        #From the graph, it appears that the two classes (benign and malignant), when projected to a two-dimensional space,
        #can be separable.

        print('Explained variation per principal component: {}'.format((pca.explained_variance_ratio_)*100))


        df = pd.DataFrame({'Cumulative Variance Explained %':np.cumsum(pca.explained_variance_ratio_)*100,
                    'No. of Components':['1','2', '3', '4']})
        sns.lineplot(x='No. of Components',y="Cumulative Variance Explained %", data=df, color="b")
        plt.title("Cumulative Variance Explained by PC\n", fontsize=20, color="b")
        plt.show()
        print(DIVIDE)

        #The graph shows that over 90% of the variance is explained by the first 3 principal components. 
        #While not a material decrease from all 4 features available in the data, it is possible to use
        #principal component analysis to decrease model run time without material decrease in performance.        
        return self.pcaData_toSplit
 
    def feature_engineering(self):

        # One-hot-encoding is applied to nominal variables to transform them so each level of the feature becomes 
        # a new column with a value of 0 or 1 assigned, based on whether the level exists for a specific record. 
        # In order to remove collinearity, one level (i.e. new column) of the transformed variable is dropped from the data

        # As there are ordinal, nominal and numeric fields, fix data to be consistent with their types
        # Apply one-hot-encoding to the nominal (catergorical) fields (Shape and Margin)
        # Need to drop one of the levels for each - drop first level
        cat_features = ['Shape', 'Margin']
        featureData = pd.get_dummies(self.mammoData, columns = cat_features, drop_first=True)
        
        # Re-order columns so target (Severity) is at the end
        self.finalData = featureData[['Age', 'Density', 'Shape_2', 'Shape_3', 'Shape_4', 'Margin_2', 'Margin_3', 'Margin_4', 'Margin_5', 'Severity']]

        print("Final Data:")
        print()
        print(self.finalData.head())

        print(DIVIDE)
        
        print("Final Data Information:")
        print()
        self.finalData.info()
        
        # Correlation Matrix
        # A heatmap is used to determine whether identify correlations between features. 
        # A stronger relationship will be represented by a darker blue shade in the matrix. 
        # If variables are highly correlated, this may add complexity without model improvement. 
        # Based on the following matrix, there does not appear to be any strong correlations between features, 
        # so there is no need to remove any from the analysis.
        
        #Create a heatmap to look for any collinearity of features
        sns.heatmap(self.finalData.corr(),cmap='Blues', annot = True,fmt='.1g');
        plt.title("Heatmap for All Features")
        plt.show()
        print(DIVIDE)
        
        self.finalData.to_csv('finalData.txt', index = False)

        return self.finalData