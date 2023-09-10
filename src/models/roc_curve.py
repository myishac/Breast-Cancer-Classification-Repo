import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve

class Roc_Curve:
    def __init__(self, list_model_class,X_test,y_test):
        self.list_model_class = list_model_class
        self.X_test = X_test
        self.y_test = y_test
        return
    
    def plot_roc_curve(self):
        y_probs = []
        r_curves = {}
        for model_class in self.list_model_class:
            y_predict= model_class.get_fit_model().predict_proba(self.X_test)[:,1]
            y_probs.append(y_predict)
            false_pos, sens, thresh = roc_curve(self.y_test, y_predict)
            auc_model = metrics.roc_auc_score(self.y_test, y_predict)
            r_curves[model_class] = [false_pos, sens, thresh, auc_model]
            
        for model in r_curves:
            plt.plot(r_curves[model][0], r_curves[model][1], label = "{} (auc = {} )".format(str(model), round(r_curves[model][3],2)))

        plt.ylabel("Sensitivity")
        plt.xlabel("False Positive Rate")
        plt.title("ROC curve for Classification Models", fontsize=16);
        plt.legend();
        plt.show()
        
        return 