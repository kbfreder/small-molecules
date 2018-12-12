import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import make_scorer, classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score


seed = 19

def assess_model(preprocessor, model, X, y, n=5):
    '''Stratified k-fold cross-validation, returns ALL THE THINGS:
    precision, recall, f1-score, confusion matrix, AUC, accuracy'''
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)

    precs = []
    recalls = []
    f1s = []
    conf_mat = []
    aucs = []
    accs = []

    for train, test in cv.split(X, y):
        pipe.fit(X.loc[train], y[train])
        y_pred = pipe.predict(X.loc[test])
        # CatBoost's predict output is float64. sklearn scoring metrics require integers
        y_pred = y_pred.astype(int)
        y_proba = pipe.predict_proba(X.loc[test])[:,1]

        precs.append(precision_score(y[test],y_pred, average=None))
        recalls.append(recall_score(y[test], y_pred, average=None))
        f1s.append(f1_score(y[test], y_pred, average=None))
        conf_mat.append(confusion_matrix(y[test], y_pred))
        aucs.append(roc_auc_score(y[test],y_proba))
        accs.append(accuracy_score(y[test], y_pred))

    # print(f1s, conf_mat)
    twos = [precs, recalls, f1s]
    data1 = [[s[i][j] for i in range(n)] for j in range(2) for s in twos]
    data1_new = np.mean(data1, axis=1).reshape(1,6)
    df1 = pd.DataFrame(data1_new,
             columns=['Precision-0', 'Recall-0 (Specificty)','F1score-0','Precision-1',
             'Recall-1 (Sensitivity)','F1score-1']).mean()
    data2 = [[s[i][j] for j in range(2) for i in range(2)] for s in conf_mat]
    df2 = pd.DataFrame(data2, columns=['TN','FN','FP','TP']).mean()
    # df3 = pd.DataFrame(aucs, columns=['AUC']).mean()
    df3 = pd.DataFrame(list(zip(aucs, accs)), columns=['AUC','Accuracy']).mean()
    return df1.append(df2.append(df3))
    # return (precs, recalls, f1s)

def assess_model_only(model, X, y, n=5):
    '''Stratified k-fold cross-validation, returns ALL THE THINGS:
    precision, recall, f1-score, confusion matrix, AUC, accuracy'''
    # pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)

    precs = []
    recalls = []
    f1s = []
    conf_mat = []
    aucs = []
    accs = []

    for train, test in cv.split(X, y):
        # y_pred = pipe.fit(X.loc[train], y[train]).predict(X.loc[test])
        # y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])[:,1]
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        y_pred = y_pred.astype(int)
        y_proba = model.predict_proba(X[test])[:,1]

        precs.append(precision_score(y[test],y_pred, average=None))
        recalls.append(recall_score(y[test], y_pred, average=None))
        f1s.append(f1_score(y[test], y_pred, average=None))
        conf_mat.append(confusion_matrix(y[test], y_pred))
        aucs.append(roc_auc_score(y[test],y_proba))
        accs.append(accuracy_score(y[test], y_pred))

def assess_model_f1(preprocessor, model, X, y, n=5, pos_label=1):
    '''Returns f1 score for positive class only. Specify if not 1'''
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = []
    for train, test in cv.split(X, y):
        y_pred = pipe.fit(X.loc[train], y[train]).predict(X.loc[test])
        # y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])[:,1]
        scores.append(f1_score(y[test], y_pred, average='binary', pos_label=pos_label))

    return np.mean(scores)

def assess_model_recall(preprocessor, model, X, y, n=5, pos_label=1):
    '''Returns recall (specifity) for positive class (specify pos_lable if it is not 1)'''
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = []
    for train, test in cv.split(X, y):
        y_pred = pipe.fit(X.loc[train], y[train]).predict(X.loc[test])
        # y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])[:,1]
        scores.append(recall_score(y[test], y_pred, average='binary', pos_label=pos_label))

    return np.mean(scores)

def assess_model_precision(preprocessor, model, X, y, n=5, pos_label=1):
    '''Returns recall (specifity) for positive class (specify pos_lable if it is not 1)'''
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = []
    for train, test in cv.split(X, y):
        y_pred = pipe.fit(X.loc[train], y[train]).predict(X.loc[test])
        scores.append(precision_score(y[test], y_pred, average='binary', pos_label=pos_label))

    return np.mean(scores)

def false_neg_scorer(y_true, y_pred):
    '''FNR = 1 - TPR = FN / (FN + TP)'''
    conf_mat_nums = confusion_matrix(y_true, y_pred).ravel()
    return 1 - (conf_mat_nums[1] / conf_mat_nums[1] + conf_mat_nums[3])

def specificity_scorer(y_true, y_pred):
    '''Specificity = TN / (TN + FP)'''
    conf_mat_nums = confusion_matrix(y_true, y_pred).ravel()
    return conf_mat_nums[0] / (conf_mat_nums[0] + conf_mat_nums[2])

def precision_neg_class(y_true, y_pred):
    '''Precision-0 = TN / (TN + FN)'''
    conf_mat_nums = confusion_matrix(y_true, y_pred).ravel()
    return conf_mat_nums[0] / (conf_mat_nums[0] + conf_mat_nums[1])

def conf_mat_avg_cv(preprocessor, model, X, y, n=5):
    '''Returns "average" confusion matrix'''
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    conf_mat = []

    for train, test in cv.split(X, y):
        y_pred = pipe.fit(X.loc[train], y[train]).predict(X.loc[test])
        conf_mat.append(confusion_matrix(y[test], y_pred))

    df_data = [[s[i][j] for j in range(2) for i in range(2)] for s in conf_mat]
    return (pd.DataFrame(df_data,
                         columns=['TN','FN','FP','TP'])).mean()

def assess_preproc_model_auc(preprocessor, model, X, y, n=5):
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = cross_validate(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1,
                            return_train_score=False)
    return np.mean(scores['test_score'])

def assess_pipe_auc(pipe, X, y, n=5):
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = cross_validate(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1,
                            return_train_score=False)
    return np.mean(scores['test_score'])

def get_avg_roc_curve(preprocessor, model, X, y, n=5):
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, _ = roc_curve(y[test], y_proba[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    return [mean_fpr, mean_tpr, mean_auc, std_auc]

def test_roc_auc(preproc, model, X_train, y_train, X_test, y_test):
    pipe = Pipeline(steps=[('preprocessor', preproc), ('classifier', model)])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    return [fpr, tpr, roc_auc]

def assess_model_with_resamp(preprocessor, model, res, X, y, n=5):
    '''Tests model with preprocessing steps, and 'res' resampler
    using Stratified k-fold cross-validation and ROC-AUC as metric'''

    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = []
    for train, test in cv.split(X, y):
        X_res, y_res = res.fit_resample(X[train], y[train])
        pipe.fit(X_res, y_res)
        y_proba = pipe.predict_proba(X[test])[:,1]
        scores.append(roc_auc_score(y[test], y_proba))

    return np.mean(scores), np.std(scores)

def assess_model_with_resamp_prec(preprocessor, model, res, X, y, n=5, pos_label=1):
    '''Tests 'model' with 'preprocessor' preprocessing pipeline,
    after applying 'res' resampler to the data X & y.
    Uses Stratified k-fold cross-validation and precision score as metric'''

    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = []
    for train, test in cv.split(X, y):
        X_res, y_res = res.fit_resample(X[train], y[train])
        pipe.fit(X_res, y_res)
        y_pred = pipe.predict(X[test])
        scores.append(precision_score(y[test], y_pred, average='binary', pos_label=pos_label))
    return np.mean(scores)

def assess_model_only_with_resamp_prec(model, res, X, y, n=5, pos_label=1):
    '''Tests model after 'res' resampler
    using Stratified k-fold cross-validation and precision as metric'''

    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = []
    for train, test in cv.split(X, y):
        X_res, y_res = res.fit_resample(X[train], y[train])
        model.fit(X_res, y_res)
        y_pred = model.predict(X[test])
        scores.append(precision_score(y[test], y_pred, average='binary', pos_label=pos_label))
    return np.mean(scores)

def assess_all_models_with_resamp(preprocessor, model_lib, res, X, y, n=5):
    '''Apply preprocessor and 'res' resampling. Test models in model_lib
    using Stratified K-Fold cross-validation (default 5-fold)
    ROC-AUC score is metric.
    Example model_lib = {'lr':LogisticRegression(solver='liblinear'),
                 'gb': GradientBoostingClassifier(),
                 'nb': GaussianNB()}
    '''
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = defaultdict(list)

    for train, test in cv.split(X, y):
        X_res, y_res = res.fit_resample(X.loc[train], y.loc[train])
        for mod_str, model in model_lib.items():
            pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            pipe.fit(X_res, y_res)
            y_proba = pipe.predict_proba(X.loc[test])[:,1]
            scores[mod_str].append(roc_auc_score(y.loc[test], y_proba))

    for k,v in scores.items():
        scores[k] = sum(v) / n

    return scores

def assess_all_models_with_resamp_prec(preprocessor, model_lib, res, X, y, n=5):
    '''Apply preprocessor and 'res' resampling. Test models in model_lib
    using Stratified K-Fold cross-validation (default 5-fold)
    F1 score is metric.
    Example model_lib = {'lr':LogisticRegression(solver='liblinear'),
                 'gb': GradientBoostingClassifier(),
                 'nb': GaussianNB()}
    '''
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = defaultdict(list)

    for train, test in cv.split(X, y):
        X_res, y_res = res.fit_resample(X.loc[train], y.loc[train])
        for mod_str, model in model_lib.items():
            pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            pipe.fit(X_res, y_res)
            # y_proba = pipe.predict_proba(X.loc[test])[:,1]
            y_pred = pipe.predict(X.loc[test])[:,1]
            # scores[mod_str].append(precision_score(y.loc[test], y_pred))
            scores.append(precision_score(y[test], y_pred, average='binary', pos_label=pos_label))

    for k,v in scores.items():
        scores[k] = sum(v) / n

    return scores

def assess_multiple_models(preprocessor, model_lib, X, y, n=5):
    '''Returns ROC-AUC
    Example model_lib = {'lr':LogisticRegression(solver='liblinear'),
                 'gb': GradientBoostingClassifier(),
                 'nb': GaussianNB()}
    '''
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = defaultdict(list)

    for train, test in cv.split(X, y):
        # X_res, y_res = res.fit_resample(X.loc[train], y.loc[train])
        for mod_str, model in model_lib.items():
            pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            pipe.fit(X.loc[train], y.loc[train])
            y_proba = pipe.predict_proba(X.loc[test])[:,1]
            scores[mod_str].append(roc_auc_score(y.loc[test], y_proba))

    for k,v in scores.items():
        scores[k] = sum(v) / n

    return scores

def get_avg_roc_curve(preprocessor, model, X, y, n=5):
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, _ = roc_curve(y[test], y_proba[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    return [mean_fpr, mean_tpr, mean_auc, std_auc]

def get_avg_npv_curve(preprocessor, model, X, y, n=5):
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n)

    fnrs, tnrs, npvs = [], [], []
#     mean_fnr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])
        fnr, tnr, npv = [], [], []

        for cutoff in cut:
            pred = np.array(y_proba[:,1] > cutoff)
            surv = y[test]
            fpos = pred * (1 - surv)
            tpos = pred * surv
            fneg = (1 - pred) * surv
            tneg = (1 - pred) * (1 - surv)

            fnr.append(np.sum(fneg) / (np.sum(fneg) + np.sum(tpos)))
            tnr.append(np.sum(tneg) / (np.sum(tneg) + np.sum(fpos)))
            npv.append(np.sum(tneg / (np.sum(tneg) + np.sum(fneg))))

        tnrs.append(tnr)
        fnrs.append(fnr)
        npvs.append(npv)

    mean_tnr = np.mean(tnrs, axis=0)
    mean_fnr = np.mean(fnrs, axis=0)
    mean_npv = np.mean(npvs, axis=0)
    npv_auc = auc(mean_fnr, mean_tnr)
    return [mean_fnr, mean_tnr, npv_auc, mean_npv]

def cost(threshold):
    y_pred = np.array(y_proba > threshold)
    fpos = y_pred * (1 - y_test)
    fneg = (1 - y_pred) * y_test

    cost = fpos * fp_cost
    revenue = tpos * tp_rev
    profit = np.sum(revenue - cost)
    return np.sum(cost)

def eval_final_model(y_test, y_pred, y_proba, classes=None):
    '''Classes are a list of class labels to print on confusion matrix'''
    conf_mat = confusion_matrix(y_test, y_pred)
    fmt = '0'
    thresh = conf_mat.max() / 2.
    if not classes:
        classes = ['Class 0', 'Class 1']

    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                     horizontalalignment="center",
                     size='16',
                     color="white" if conf_mat[i, j] > thresh else "black")

    precs = precision_score(y_test,y_pred, average=None)
    recalls = recall_score(y_test, y_pred, average=None)
    f1s = f1_score(y_test, y_pred, average=None)
    aucs = roc_auc_score(y_test,y_proba)
    accs = accuracy_score(y_test, y_pred)

    twos = [precs, recalls, f1s]
    data1 = [[s[j]] for j in range(2) for s in twos]
    data1_new = np.mean(data1, axis=1).reshape(1,6)
    df1 = pd.DataFrame(data1_new,
             columns=['Precision-0', 'Recall-0 (Specificty)','F1score-0','Precision-1',
             'Recall-1 (Sensitivity)','F1score-1']).mean()
    data2 = conf_mat.flatten().reshape(-1,4)
    df2 = pd.DataFrame(data2, columns=['TN','FP','FN','TP']).mean()
    data3 = np.array([aucs, accs]).reshape(-1, 2)
    df3 = pd.DataFrame(data3, columns=['AUC','Accuracy']).mean()

    df = df1.append(df2.append(df3))
    print(df)
    return df
