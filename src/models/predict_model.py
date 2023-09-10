from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


class Predict_Model:
    def __init__(self, model_class, X_test, y_test):
        self.model_class = model_class
        self.model = model_class.get_model()
        self.X_test = X_test
        self.y_test = y_test

    def predictions(self):
        return self.model.predict(self.X_test)

    def confMatrix(self):
        y_predict = self.predictions()
        print(confusion_matrix(self.y_test, y_predict))
        print('Classification Report for',str(self.model_class),'\n',classification_report(self.y_test, y_predict))
        return
        
    #Calculate metrics for each model using test data:
    def calcMetrics(self):
        y_pred = self.predictions()
        tn, fp, fn, tp = confusion_matrix(list(self.y_test), list(y_pred), labels=[0, 1]).ravel()
        acc = round((tp+tn)/(tp+tn+fp+fn) *100,2)
        prec = round((tp)/(tp+fp) *100,2)
        far = round((fp)/(fp+tn) *100,2)
        fnr = round(fn/(fn+tp) *100,2)
        return acc, prec, far, fnr
