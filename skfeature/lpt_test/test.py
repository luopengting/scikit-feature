import scipy.io
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import KFold
from skfeature.utility.sparse_learning import *
from skfeature.function.sparse_learning_based import ll_l21


def main():
    # load data
    # mat = scipy.io.loadmat('../data/COIL20.mat')
    # X = mat['X']    # data
    # X = X.astype(float)
    # y = mat['Y']    # label
    # print(y)
    # print(y.shape)
    # y = y[:, 0]
    # print(y)
    # print(y.shape)

    sequence_name = 'D3'
    feature_model_num = 0

    img_feature_path = '/usr/luopengting/shareHoldersWithGPU_a/luopengting/workplace/python/pytorch/breast_cancer_lymph/data/' \
                       + sequence_name + '/model_' + str(feature_model_num) + '_img_feature.txt'
    img_label_path = '/usr/luopengting/shareHoldersWithGPU_a/luopengting/workplace/python/pytorch/breast_cancer_lymph/data/' \
                       + sequence_name + '/labels.txt'
    train = pd.read_csv(img_feature_path, sep=' ', header=None)
    train_labels = pd.read_csv(img_label_path, sep='\t', header=None)

    X_train = train.as_matrix(columns=None)
    y_train = train_labels.as_matrix(columns=None)
    # y.astype(int)
    y_train = y_train[:, 0]
    # print(y)

    ss = KFold(n_splits=5)
    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = 0
    acc_total = 0
    for train, test in ss.split(X_train):
        X = X_train[train]
        y = y_train[train]

        # obtain the feature weight matrix
        Weight, obj, value_gamma = ll_l21.proximal_gradient_descent(X, y, 0.1, verbose=False)

        # sort the feature scores in an ascending order according to the feature scores
        idx = feature_ranking(Weight)

        # perform evaluation on clustering task
        num_fea = 100  # number of selected features

        # obtain the dataset on the selected features
        selected_features = X[:, idx[0:num_fea]]
        classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                        intercept_scaling=1, class_weight='balanced', random_state=None, solver='liblinear',
                                        max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
        classifier.fit(selected_features, y)

        X_test = X_train[test]
        X_test = X_test[:, idx[0:num_fea]]
        Y_test = y_train[test]
        Y_pred = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(Y_test, Y_pred)
        print(confusion_matrix)

        # # ROC
        # from sklearn.metrics import roc_auc_score, auc
        # from sklearn import metrics
        # import matplotlib.pyplot as plt
        # y_predict = classifier.predict(X_test)
        # y_probs = classifier.predict_proba(X_test)  # 模型的预测得分
        # # print(y_probs)
        # fpr, tpr, thresholds = metrics.roc_curve(list(Y_test.reshape((Y_test.shape[0], 1))), list(y_probs[:, 1]),
        #                                          pos_label=1)
        # roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
        # # 开始画ROC曲线
        # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([-0.1, 1.1])
        # plt.ylim([-0.1, 1.1])
        # plt.xlabel('False Positive Rate')  # 横坐标是fpr
        # plt.ylabel('True Positive Rate')  # 纵坐标是tpr
        # plt.title('Receiver operating characteristic example')
        # plt.show()

    # output the average NMI and average ACC
    # print('NMI:', float(nmi_total)/20)
    # print('ACC:', float(acc_total)/20)

if __name__ == '__main__':
    main()