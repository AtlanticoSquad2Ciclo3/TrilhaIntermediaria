from sklearn.model_selection import train_test_split
from sklearn import metrics,svm
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from sklearn.preprocessing import MaxAbsScaler
import time
import pandas as pd

from pipeline_utils import Pipeline1
def generate_svm_model(train_data,label_train_data,test_data):
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data, label_train_data)
    resultado = clf.predict(test_data)
    return resultado
def generate_SGDC_model(train_data,label_train_data,test_data):
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=200)
    clf.fit(train_data, label_train_data)
    resultado = clf.predict(test_data)
    return resultado
def generate_naive_bayes_model(train_data,label_train_data,test_data):
    gnb = GaussianNB()
    gnb.fit(train_data, label_train_data)
    resultado = gnb.predict(test_data)
    return resultado
def generate_decision_tree_model(train_data,label_train_data,test_data):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, label_train_data)
    resultado = clf.predict(test_data)
    return resultado
def generate_random_forest_model(X_train, y_train,test_data):
    rfc = RandomForestClassifier(criterion= 'entropy', max_depth= 8, max_features='auto', n_estimators=200)
    rfc.fit(X_train,y_train)
    resultado = rfc.predict(test_data)
    return resultado
def generate_MLP_model(X_train, y_train,test_data):
    classifier = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    classifier.fit(X_train, y_train)
    resultado = classifier.predict(test_data)
    return resultado
def generate_knn_model(train_data,label_train_data,test_data):
    knn = KNeighborsClassifier()
    knn.fit(train_data,label_train_data)
    resultado = knn.predict(test_data)
    return resultado
def gen_classifiers(train_data,label_train_data,test_data):
    resultados = {}
    tempos = []
    
    for gmodel,model_name in zip([generate_knn_model,
                   generate_MLP_model,                             
                   generate_SGDC_model,
                   generate_svm_model,
                   generate_decision_tree_model,
                   generate_naive_bayes_model,
                   generate_random_forest_model,],['knn','mlp','sgdc','svm','dtm','nbm','rfm']):
        start = time.time()
        resultados[model_name] = gmodel(train_data,label_train_data,test_data)
        end = time.time()
        tempos.append(end-start)
    return resultados,tempos
def get_metrics(resultados,tempos, y_test):
    dict_out = {}
    for resultado,tempo in zip(resultados.items(),tempos):
        algo,resultado = resultado
        acc = metrics.accuracy_score(y_test,resultado)
        recall = metrics.recall_score(y_test,resultado,average='weighted')
        precision = metrics.precision_score(y_test,resultado,average='weighted')
        f1 = metrics.f1_score(y_test,resultado,average='weighted')
        dict_out[algo] = {"acc":acc,
                          "recall":recall,
                          "precision":precision,
                          "f1":f1,
                          'time':tempo}

    
    return pd.DataFrame(dict_out).transpose()  

def baseline( X_train,X_test,y_train,y_test):
    X_train_1 = np.stack(X_train.copy().apply(lambda x: x.reshape(-1)).values).astype(np.float32)/255
    X_test_1  = np.stack(X_test.copy().apply(lambda x: x.reshape(-1)).values).astype(np.float32)/255
    resultados,tempos = gen_classifiers(X_train_1, y_train, X_test_1)
    return X_train_1,X_test_1,get_metrics(resultados,tempos, y_test)

def get_contours_param(contour):
    contour_area, contour_perimeter, contour_convex_area, eccentricity = 0, 0, 0, 0
    max_area = 0
    for c in contour:
      contour_area += c.filled_area
      contour_perimeter += c.perimeter
      contour_convex_area += c.convex_area
      if c.filled_area >= max_area:
        eccentricity = c.eccentricity
        max_area = c.filled_area
    return contour_area, contour_perimeter, contour_convex_area, eccentricity

def _feature_extraction_pipeline1(img):
    regions = regionprops(np.uint8(img))
    area, perimeter, convex_area, eccentricity = get_contours_param(regions)
    return area, perimeter, convex_area, eccentricity

def Pipeline1Classifier( X_train,X_test,y_train,y_test):
    features = ['area', 'perimeter', 'convex_area', 'eccentricity']
    transform = Pipeline1().transform
    X_train_1 = X_train.copy().apply(lambda x: transform(x)['otsuFilter']).apply(_feature_extraction_pipeline1)
    X_test_1  = X_test.copy().apply(lambda x: transform(x)['otsuFilter']).apply(_feature_extraction_pipeline1)
    
    X_train_1 = X_train_1.apply(pd.Series)
    X_train_1.columns = features
    
    X_test_1 = X_test_1.apply(pd.Series)
    X_test_1.columns = features
    
    norm = MaxAbsScaler()
    X_train_1 = norm.fit_transform(X_train_1)
    X_test_1 = norm.fit_transform(X_test_1)
    
    
    resultados,tempos = gen_classifiers(X_train_1, y_train, X_test_1)
    return X_train_1,X_test_1, get_metrics(resultados,tempos,y_test)
    
    print
if __name__ == "__main__":
    from pipeline_utils import ingestao
    import numpy as np
    p = "/home/eduardo/Downloads/projetos/classificacao_plantas"
    dados = ingestao(p,resize=1.0)  
    X, y = dados['img'], dados['y_true']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, stratify=y, random_state=1337)
    
    X_train_1,X_test_1,out = Pipeline1Classifier(X_train,X_test,y_train,y_test)
    
    print(out)