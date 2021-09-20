# -*- coding: utf-8 -*-
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge

import json

def allclf():
    classifierList = []
    
    classifierList.append((linear_model.LinearRegression(), "LinearRegression", "predictONLY"))  # NO random_state!!!
    classifierList.append((linear_model.Ridge(random_state=RND_init), "Ridge", "predictONLY"))

    classifierList.append((linear_model.Lasso(random_state=RND_init), "Lasso", "predictONLY"))
    
    #classifierList.append((linear_model.MultiTaskLasso(), "linear_model.MultiTaskLasso", "predictONLY"))     # ValueError: For mono-task outputs, use ElasticNet
    
    classifierList.append((linear_model.ElasticNet(max_iter=10000, random_state=RND_init), "ElasticNet", "predictONLY")) # ValueError: Can't handle mix of binary and continuous
    
    #classifierList.append((linear_model.MultiTaskElasticNet(), "linear_model.MultiTaskElasticNet", "predictONLY"))   # ValueError: For mono-task outputs, use ElasticNet
    
    # куча варнингов и бред в результатах
    # classifierList.append((linear_model.Lars(), "linear_model.Lars", "predictONLY"))    # ConvergenceWarning: Regressors in active set degenerate. Dropping a regressor, after 3 iterations, i.e. alpha=1.662e-02, with an active set of 3 regressors, and the smallest cholesky pivot element being 2.220e-16. Reduce max_iter or increase eps parameters.
    
    classifierList.append((linear_model.LassoLars(), "LassoLars", "predictONLY"))  # NO random_state!!!
    classifierList.append((linear_model.OrthogonalMatchingPursuit(), "OrthogonalMatchingPursuit", "predictONLY"))  # NO random_state!!!
    classifierList.append((linear_model.BayesianRidge(), "BayesianRidge", "predictONLY"))  # NO random_state!!!
    classifierList.append((linear_model.HuberRegressor(), "HuberRegressor", "predictONLY"))  # NO random_state!!!
    
    #classifierList.append((linear_model.RANSACRegressor(), "linear_model.RANSACRegressor", "predictONLY"))   # ValueError: min_samples may not be larger than number of samples X.shape[0].
    classifierList.append((linear_model.TheilSenRegressor(), "TheilSenRegressor", "predictONLY"))

    ### TODO : 1.1.16. Polynomial regression:
    #classifierList.append((discriminant_analysis.LinearDiscriminantAnalysis(), "discriminant_analysis.LinearDiscriminantAnalysis", "predict_proba")) # ValueError: Unknown label type: (array([8.0294, 8.4   , 8.2833, 8.25  , 8.1315, 8.0007, 7.8348, 8.1709, 8.1725, 8.635 , 8.1616, 7.419 , 8.0138, 8.108 ]),)
    #classifierList.append((discriminant_analysis.QuadraticDiscriminantAnalysis(), "discriminant_analysis.QuadraticDiscriminantAnalysis", "predict_proba")) # ValueError: Unknown label type: 'continuous'
    
    classifierList.append((KernelRidge(), "KernelRidge", "predictONLY"))  # NO random_state!!!
    from sklearn.cross_decomposition import PLSCanonical
    ### TODO ПРОВЕРИТЬ - вылетает при форматировании!!!
    # classifierList.append((PLSCanonical(), "PLSCanonical", "predictONLY"))  # TypeError: unsupported format string passed to numpy.ndarray.__format__
    
    from sklearn.cross_decomposition import PLSRegression
    ### TODO ПРОВЕРИТЬ - вылетает при форматировании!!!
    # classifierList.append((PLSRegression(), "PLSRegression", "predictONLY"))  # TypeError: unsupported format string passed to numpy.ndarray.__format__
    #classifierList.append((linear_model.SGDClassifier(random_state=RND_init), "linear_model.SGDClassifier", "decision_function"))   # ValueError: Unknown label type: (array([7.419 , 7.8348, 8.0007, 8.0138, 8.0294, 8.108 , 8.1315, 8.1616, 8.1709, 8.1725, 8.25  , 8.2833, 8.4   , 8.635 ]),)
    #classifierList.append((linear_model.Perceptron(random_state=RND_init), "linear_model.Perceptron", "decision_function")) # ValueError: Unknown label type: (array([7.419 , 7.8348, 8.0007, 8.0138, 8.0294, 8.108 , 8.1315, 8.1616, 8.1709, 8.1725, 8.25  , 8.2833, 8.4   , 8.635 ]),)
    #classifierList.append((linear_model.PassiveAggressiveClassifier(loss='hinge', random_state=RND_init), "linear_model.PassiveAggressiveClassifier(loss='hinge')", "decision_function")) # ValueError: Unknown label type: (array([7.419 , 7.8348, 8.0007, 8.0138, 8.0294, 8.108 , 8.1315, 8.1616, 8.1709, 8.1725, 8.25  , 8.2833, 8.4   , 8.635 ]),)
    #classifierList.append((linear_model.PassiveAggressiveClassifier(loss='squared_hinge', random_state=RND_init), "linear_model.PassiveAggressiveClassifier(loss='squared_hinge')", "decision_function")) # ValueError: Unknown label type: (array([7.419 , 7.8348, 8.0007, 8.0138, 8.0294, 8.108 , 8.1315, 8.1616, 8.1709, 8.1725, 8.25  , 8.2833, 8.4   , 8.635 ]),)
    #classifierList.append((LinearSVC(random_state=RND_init), "LinearSVC", "decision_function")) # ValueError: Unknown label type: 'continuous'
    #classifierList.append((NuSVC(nu=0.1, random_state=RND_init), "NuSVC(nu=0.1)", "decision_function")) # ValueError: Unknown label type: 'continuous'
    #classifierList.append((NuSVC(nu=0.3, random_state=RND_init), "NuSVC(nu=0.3)", "decision_function")) # ValueError: Unknown label type: 'continuous'
    
    classifierList.append((linear_model.ARDRegression(), "ARDRegression", "predictONLY")) # долго работает (50 сек)
    
    # classifierList.append((linear_model.LogisticRegression(random_state=RND_init), "linear_model.LogisticRegression", "predict_proba")) # ValueError: Unknown label type: 'continuous'
    #classifierList.append((GaussianProcessClassifier(random_state=RND_init), "GaussianProcessClassifier", "predict_proba")) # ValueError: Unknown label type: (array([8.0294, 8.4   , 8.2833, 8.25  , 8.1315, 8.0007, 7.8348, 8.1709, 8.1725, 8.635 , 8.1616, 7.419 , 8.0138, 8.108 ]),)
    from sklearn.naive_bayes import GaussianNB
    #classifierList.append((GaussianNB(), "GaussianNB (Gaussian Naive Bayes)", "predict_proba"))  # ValueError: Unknown label type: (array([7.419 , 7.8348, 8.0007, 8.0138, 8.0294, 8.108 , 8.1315, 8.1616, 8.1709, 8.1725, 8.25  , 8.2833, 8.4   , 8.635 ]),)
    from sklearn import tree
    # classifierList.append((tree.DecisionTreeClassifier(random_state=RND_init), "tree.DecisionTreeClassifier", "predict_proba")) # ValueError: Unknown label type: 'continuous'
    # classifierList.append((KNeighborsClassifier(n_neighbors=5), "KNeighborsClassifier(n_neighbors=5)", "predict_proba"))     # ValueError: Unknown label type: 'continuous'
    
    from sklearn import neural_network
    # classifierList.append((neural_network.MLPClassifier(random_state=RND_init), "neural_network.MLPClassifier", "predict_proba")) # ValueError: Unknown label type: (array([8.0294, 8.4   , 8.2833, 8.25  , 8.1315, 8.0007, 7.8348, 8.1709, 8.1725, 8.635 , 8.1616, 7.419 , 8.0138, 8.108 ]),)


    target_interval = 0.14
    min_syndr_on_count = 1  # 2
    fea_to_use = X_learn.shape[1]
    clf = sand.SimpleSyndromeRegressor()
    #clf = sand.SimpleSyndromeRegressor(max_features=fea_to_use, #  - 5,
    #                                   min_sympt_on_count=2,  # 2,
    #                                   max_syndr_on_count=15,  # 5
    #                                   min_syndr_on_count=min_syndr_on_count,
    #                                   # estimation='distance',
    #                                   target_interval=target_interval)
    classifierList.append((clf, "SAND_SimpleSyndromeRegressor", "predictONLY"))

# random_state=15 Почему? Из-за анекдота:
#Петька с Василь Иванычем угнали самолет. Летят. Вдруг Василь Иваныч орет:
#– Петька, приборы!
#– 15!
#– Что "15"?!
#– А что "приборы"?


def procTaskSimple():
    classifier = linear_model.LinearRegression()
    print(f"classifier = {classifier}")  # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    classifier2 = linear_model.Ridge(random_state=15)
    print(f"classifier2 = {classifier2}")  # Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=15, solver='auto', tol=0.001)
    classifier3 = linear_model.Ridge()
    #classifier3.random_state = 15
    print(f"classifier3 = {classifier3}")  # Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
    print(type(classifier3).__name__)   # Ridge
    settingsJson = '{"random_state": 15, "random_state2": "15"}'
    json_object = json.loads(settingsJson)
    print("=== BEGIN LOOP ===")
    for key, value in json_object.items():
        print(f" 0. {key} = {value}")
        print('   typeof(value)=' + type(value).__name__)
        if type(value).__name__== 'str':
            exec(f'classifier3.{key} = ''{value}''')
        else:
            exec(f'classifier3.{key} = {value}')
        print(f" 1. {key} = {value}")
    print(classifier3)
    print("=== END LOOP ===")



def procTask():
    classifierList = []
    classifier = linear_model.LinearRegression()
    print(f"classifier = {classifier}")  # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    classifier2 = linear_model.Ridge(random_state=15)
    print(f"classifier2 = {classifier2}")  # Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=15, solver='auto', tol=0.001)
    classifier3 = linear_model.Ridge()
    #classifier3.random_state = 15
    print(f"classifier3 = {classifier3}")  # Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
    print(type(classifier3).__name__)   # Ridge


    folderPath = sys.argv[1] #"D:\\MyProjects\\Python\\2019_realValue4AspNetCore\\DataFiles"
    # Путь к файлам с исходными данными
    sfilePathTrainExcel = sys.argv[2]
    sfilePathPredictExcel = sys.argv[3] 
    timeout4Method = 0
    if len(sys.argv)>4:
        timeout4Method = int(sys.argv[4])


    settingsJson = '''{
    "folderPath": "D:\MyProjects\Python\2019_realValue4AspNetCore\DataFiles",
    "fileTrain": "TrainingSet.xls",
    "filePredict": "Prediction.xls",
    "timeout4Method": 0,
    "algorithms": [
        {"name": "LinearRegression", "mode": "predictONLY", "class": "linear_model.LinearRegression()", "settings": {}},
        {"name": "Ridge", "mode": "predictONLY", "class": "linear_model.Ridge()", "settings": {"random_state": 15, "random_state2": "15"}}
    ]}'''
    json_object = json.loads(settingsJson)
    print("=== BEGIN LOOP ===")
    for algorithm in json_object["algorithms"]:
        classifier = None
        print(f" algorithm.name = {algorithm['name']}, algorithm.class = {algorithm['class']}")
        print(f"classifier = {algorithm['class']}")
        #classifier = get_class("algorithm['class']")
        classifier = eval(algorithm['class'])
        #exec(f"classifier = {algorithm['class']}")
        print(f"type(classifier) = {type(classifier).__name__}")
        for key, value in algorithm["settings"].items():
            print(f" 0. {key} = {value}")
            print('   typeof(value)=' + type(value).__name__)
            if type(value).__name__== 'str':
                exec(f'classifier.{key} = ''{value}''')
            else:
                exec(f'classifier.{key} = {value}')
            print(f" 1. {key} = {value}")
        print(f"classifier = {classifier}")

        classifierList.append(classifier)
    print("=== END LOOP ===")


if __name__ == '__main__':
    procTask()

