import numpy as np
from sklearn.svm import SVC
import joblib #from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import preprocessing as skpp
import time
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imbalanced_ensemble.ensemble import RUSBoostClassifier
from imbalanced_ensemble.ensemble import EasyEnsembleClassifier
from imbalanced_ensemble.ensemble import BalanceCascadeClassifier
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from pylab import mpl


def show_confusion_matrix(truelabel, predictlabel, titlename, figname):
    # compute evaluation metrics
    matrix = metrics.confusion_matrix(truelabel, predictlabel)

    # plot and save confusion matrix heat map
    fig = plt.figure()  # figsize=(2, 3), dpi=1200
    ax1 = fig.add_subplot(1, 1, 1)
    h = sns.heatmap(matrix,
                    cmap="coolwarm",
                    linecolor='white',
                    linewidths=1,
                    xticklabels=['normal', 'fault1', 'fault2', 'fault3', 'fault4', 'fault5'],
                    yticklabels=['normal', 'fault1', 'fault2', 'fault3', 'fault4', 'fault5'],
                    ax=ax1,
                    cbar=False,
                    annot=True,
                    annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'},
                    fmt="d")
    cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
    cb.ax.tick_params(labelsize=12, labelcolor='black')  # 设置colorbar刻度字体大小。
    ax1.set_xticklabels(['normal', 'fault1', 'fault2', 'fault3', 'fault4', 'fault5'],
                        rotation=0, fontsize=12, color='black')
    ax1.set_yticklabels(['normal', 'fault1', 'fault2', 'fault3', 'fault4', 'fault5'],
                        rotation=0, fontsize=12, color='black')
    #  ax.tick_params(labelsize=10,color='black')
    plt.rcParams['savefig.dpi'] = 1800  # 图片像素
    plt.rcParams['figure.dpi'] = 1800  # 分辨率

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             'color': 'black'
             }
    plt.title(titlename, fontsize=12)
    plt.ylabel("True Label", font2)
    plt.xlabel("Predicted Label", font2)
    fig.savefig('./figures/' + figname, dpi=1800, bbox_inches='tight')  # 指定分辨率和路径保存
    fig.savefig('./figures/' + figname + '.svg', dpi=1800, bbox_inches='tight')  # 指定分辨率和路径保存
    

def result_log(acctest,  testmatrix, txtfolder, filename):
    with open(os.path.join(txtfolder, filename), 'w') as log:
        log.write('acc test:{:.4f}\n'.format(acctest))
        log.write('test matrix\n')
        log.write(str(testmatrix))
        log.write('\n')


def compute_auc(ytest, ytestpre):
    n_classes = len(np.unique(ytest))
    Ytrue = label_binarize(ytest, classes=list(range(n_classes)))
    Ypred = label_binarize(ytestpre, classes=list(range(n_classes)))
    auc_macro = metrics.roc_auc_score(Ytrue, Ypred, average='macro')
    auc_micro = metrics.roc_auc_score(Ytrue, Ypred, average='micro')
    return auc_macro, auc_micro    


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values 
    #0.3 * np.sin(np.pi / 2 * (1 - p) ** 2) + np.tan(np.pi / 4 * (1 - p) ** 2) * 0.7
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)
 
    
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(5, 20)
        self.output = nn.Linear(20, 6)

    def forward(self, inputs):
        x = torch.relu(self.hidden(inputs))
        output = self.output(x)
        return output


def svm_org(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    start = time.perf_counter()
    Xtr, Xval, Ytr, Yval = train_test_split(xntrain, ytrain, test_size=0.1, random_state=32, stratify=ytrain)
    svcOrg = SVC(kernel='rbf', C=1, gamma=10)
    svcOrg.fit(Xtr, Ytr.ravel())
    acctrain0 = np.sum(svcOrg.predict(Xtr) == np.squeeze(Ytr)) / Ytr.shape[0]
    accval = np.sum(svcOrg.predict(Xval) == np.squeeze(Yval)) / Yval.shape[0]
    svcOrg.fit(xntrain, ytrain.ravel())
    joblib.dump(svcOrg, modelFolder + 'svcOrg.pkl')
    ytrainpre = svcOrg.predict(xntrain)
    ytestpre = svcOrg.predict(xntest)
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svcOrg.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svcOrg')


def svm_bsmote(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    start = time.perf_counter()
    bsmo = BorderlineSMOTE(random_state=62, k_neighbors=5, kind='borderline-1')
    xntrain_smote, ytrain_smote = bsmo.fit_resample(xntrain, ytrain)
    # cbest, gbest, score_max = cv5fold(xntrain_smote, ytrain_smote)
    Xtr, Xval, Ytr, Yval = train_test_split(xntrain_smote, ytrain_smote, test_size=0.1, random_state=32,
                                            stratify=ytrain_smote)
    svc = SVC(kernel='rbf', C=1, gamma=10)
    svc.fit(Xtr, Ytr.ravel())
    acctrain0 = np.sum(svc.predict(Xtr) == np.squeeze(Ytr)) / Ytr.shape[0]
    accval = np.sum(svc.predict(Xval) == np.squeeze(Yval)) / Yval.shape[0]
    svc.fit(xntrain_smote, ytrain_smote.ravel())
    joblib.dump(svc, modelFolder + 'svcbSmo.pkl')
    ytrainpre = svc.predict(xntrain)
    ytestpre = svc.predict(xntest)
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)    
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svcbSmo.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svcbSmo')


def svm_smote(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    start = time.perf_counter()
    smo = SMOTE(random_state=62, k_neighbors=5)
    xntrain_smote, ytrain_smote = smo.fit_resample(xntrain, ytrain)
    Xtr, Xval, Ytr, Yval = train_test_split(xntrain_smote, ytrain_smote, test_size=0.1, random_state=32,
                                            stratify=ytrain_smote)
    svc = SVC(kernel='rbf', C=1, gamma=10)
    svc.fit(Xtr, Ytr.ravel())
    acctrain0 = np.sum(svc.predict(Xtr) == np.squeeze(Ytr)) / Ytr.shape[0]
    accval = np.sum(svc.predict(Xval) == np.squeeze(Yval)) / Yval.shape[0]
    svc.fit(xntrain_smote, ytrain_smote.ravel())
    joblib.dump(svc, modelFolder + 'svcSmo.pkl')
    ytrainpre = svc.predict(xntrain)
    ytestpre = svc.predict(xntest)
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svcSmo.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svcSmo')


def svm_ros(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    start = time.perf_counter()
    smo = RandomOverSampler(random_state=62)
    xntrain_smote, ytrain_smote = smo.fit_resample(xntrain, ytrain)
    Xtr, Xval, Ytr, Yval = train_test_split(xntrain_smote, ytrain_smote, test_size=0.1, random_state=32,
                                            stratify=ytrain_smote)
    svc = SVC(kernel='rbf', C=1, gamma=10)
    svc.fit(Xtr, Ytr.ravel())
    acctrain0 = np.sum(svc.predict(Xtr) == np.squeeze(Ytr)) / Ytr.shape[0]
    accval = np.sum(svc.predict(Xval) == np.squeeze(Yval)) / Yval.shape[0]
    svc.fit(xntrain_smote, ytrain_smote.ravel())
    joblib.dump(svc, modelFolder + 'svcros.pkl')
    ytrainpre = svc.predict(xntrain)
    ytestpre = svc.predict(xntest)
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)    
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svcros.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svcros')


def svm_rus(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    # random under sampling
    start = time.perf_counter()
    rus = RandomUnderSampler(random_state=62)
    xntrain_rus, ytrain_rus = rus.fit_resample(xntrain, ytrain)
    Xtr, Xval, Ytr, Yval = train_test_split(xntrain_rus, ytrain_rus, test_size=0.1, random_state=32,
                                            stratify=ytrain_rus)
    svc = SVC(kernel='rbf', C=1, gamma=10)
    svc.fit(Xtr, Ytr.ravel())
    acctrain0 = np.sum(svc.predict(Xtr) == np.squeeze(Ytr)) / Ytr.shape[0]
    accval = np.sum(svc.predict(Xval) == np.squeeze(Yval)) / Yval.shape[0]
    svc.fit(xntrain_rus, ytrain_rus.ravel())
    joblib.dump(svc, modelFolder + 'svcrus.pkl')
    ytrainpre = svc.predict(xntrain)
    ytestpre = svc.predict(xntest)
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)    
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svcrus.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svcrus')


def svm_nearmiss(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    # random under sampling
    start = time.perf_counter()
    rus = NearMiss(version=3)
    xntrain_rus, ytrain_rus = rus.fit_resample(xntrain, ytrain)
    Xtr, Xval, Ytr, Yval = train_test_split(xntrain_rus, ytrain_rus, test_size=0.1, random_state=32,
                                            stratify=ytrain_rus)
    svc = SVC(kernel='rbf', C=10, gamma=10)
    svc.fit(Xtr, Ytr.ravel())
    acctrain0 = np.sum(svc.predict(Xtr) == np.squeeze(Ytr)) / Ytr.shape[0]
    accval = np.sum(svc.predict(Xval) == np.squeeze(Yval)) / Yval.shape[0]
    svc.fit(xntrain_rus, ytrain_rus.ravel())
    joblib.dump(svc, modelFolder + 'svcnearmiss.pkl')
    ytrainpre = svc.predict(xntrain)
    ytestpre = svc.predict(xntest)
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)    
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svcnearmiss.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svcnearmiss')


def svm_RUSBoost(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    start = time.perf_counter()
    init_kwargs = {'base_estimator': SVC(kernel='rbf', C=1, gamma=10, probability=True),
                   'random_state': 62, 'n_estimators': 10}
    clf = RUSBoostClassifier(**init_kwargs)
    clf.fit(xntrain, ytrain)
    joblib.dump(clf, modelFolder + 'svm_RUSBoost.pkl')
    ytestpre = clf.predict(xntest) 
    ytrainpre = clf.predict(xntrain) 
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svm_RUSBoost.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svm_RUSBoost')
    
    
def svm_EasyEnsemble(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    start = time.perf_counter()
    init_kwargs = {'base_estimator': SVC(kernel='rbf', C=1, gamma=10),
                   'random_state': 62, 'n_estimators': 8}
    clf = EasyEnsembleClassifier(**init_kwargs)
    clf.fit(xntrain, ytrain)
    joblib.dump(clf, modelFolder + 'svm_EasyEnsemble.pkl')
    ytrainpre=clf.predict(xntrain)
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    ytestpre = clf.predict(xntest) 
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svm_EasyEnsemble.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svm_EasyEnsemble')

    
def svm_BalanceCascade(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    start = time.perf_counter()
    init_kwargs = {'base_estimator': SVC(kernel='rbf', C=1, gamma=10, probability=True),
                   'random_state': 62, 'n_estimators': 100}
    clf = BalanceCascadeClassifier(**init_kwargs)
    clf.fit(xntrain, ytrain)
    joblib.dump(clf, modelFolder + 'svm_BalanceCascade.pkl')
    ytrainpre=clf.predict(xntrain)
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    ytestpre = clf.predict(xntest) 
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svm_BalanceCascade.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svm_BalanceCascade')


def svm_weight(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    # random under sampling
    start = time.perf_counter()
    Xtr, Xval, Ytr, Yval = train_test_split(xntrain, ytrain, test_size=0.1, random_state=32, stratify=ytrain)
    svc = SVC(kernel='rbf', C=1, gamma=10, class_weight='balanced')
    svc.fit(Xtr, Ytr.ravel())
    acctrain0 = np.sum(svc.predict(Xtr) == np.squeeze(Ytr)) / Ytr.shape[0]
    accval = np.sum(svc.predict(Xval) == np.squeeze(Yval)) / Yval.shape[0]
    svc.fit(xntrain, ytrain.ravel())
    joblib.dump(svc, modelFolder + 'svcWeight.pkl')
    ytrainpre = svc.predict(xntrain)
    ytestpre = svc.predict(xntest)
    acctrain = np.sum(ytrainpre == np.squeeze(ytrain)) / ytrain.shape[0]
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    trainmatrix = metrics.confusion_matrix(ytrain, ytrainpre)
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)    
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    print('acc train', acctrain)
    print('acc test', acctest)
    print('test confusion matrix\n', testmatrix)
    result_log(acctest, testmatrix, txtfolder, 'svcWeight.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'svcWeight')


def mlp_focal_loss(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder):
    trainnor_x = torch.from_numpy(xntrain).type(torch.float32)
    trainnor_y = torch.from_numpy(np.squeeze(ytrain)).type(torch.int64)
    testnor_x = torch.from_numpy(xntest).type(torch.float32)
    testnor_y = torch.from_numpy(np.squeeze(ytest)).type(torch.int64)
    train_torch_dataset = torch.utils.data.TensorDataset(trainnor_x, trainnor_y)  # tuple (x, y)
    trainloader = torch.utils.data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0)  # for training
    trainloader1 = torch.utils.data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=1)  # for evaluating training results
    test_torch_dataset = torch.utils.data.TensorDataset(testnor_x, testnor_y)  # tuple (x, y)
    testloader = torch.utils.data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=1)  # for evaluating test results

    # establish network, train network  and save trained network
    model = Network()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = FocalLoss(weight=torch.tensor([1.,1.,1.,1.,2.,1.]), gamma=2)#1.,1.,1.,1.,2.,1.]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.005,
                                 weight_decay=0)
    # train
    n_epoch = 200
    acc_all = []
    for epoch in range(n_epoch):
        loss_epoch = []  
        for step, (batch_x, batch_y) in enumerate(trainloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss_epoch.append(loss.item())
            # print loss
            if step % 200 == 0:
                print("Epoch: ", epoch, "| Step: ", step,
                      "|loss: ", loss.item())
                # back propagation
            optimizer.zero_grad()
            loss.backward()  # compute grad
            optimizer.step()  # update parameters
        loss_epochm = np.mean(loss_epoch)
        prediction1 = model(testnor_x)
        ytestpre = torch.max(prediction1, 1)[1].numpy()
        acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
        acc_all.append(acctest)
        print("Epoch: ", epoch, "| mean loss: ", loss_epochm,'|acc:', acctest)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, './model/annFocalLoss.pth')    
    # test
    model2 = Network()
    checkpoint = torch.load('./model/annFocalLoss.pth', map_location='cpu')
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.eval()              
    with torch.no_grad():
        prediction2 = model2(testnor_x)
    ytestpre = torch.max(prediction2, 1)[1].numpy()
    acctest = np.sum(ytestpre == np.squeeze(ytest)) / ytest.shape[0]
    testmatrix = metrics.confusion_matrix(ytest, ytestpre)
    print('acc:', acctest)
    print(testmatrix)
    result_log(acctest, testmatrix, './metrics_result', 'annFocalLoss.txt')
    print('程序运行完成，已保存运行结果')
    show_confusion_matrix(ytest, ytestpre, '', 'annFocalLoss')
 
       
if __name__ == '__main__':
    method_list = ['不处理', 
                   '随机过采样', 'SMOTE过采样', 'BorderLine SMOTE 过采样',
                   'NearMiss欠采样', '随机欠采样', 'RUSBoost', 'EasyEnsemble', 'BalanceCascade',
                   '加权支持向量机', '焦点损失']
    
    if not os.path.exists('./metrics_result'):
        os.mkdir('./metrics_result')
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    modelFolder = './model/'
    txtfolder = './metrics_result/' 

    # load data
    xtrain = pd.read_excel('./data/traininput.xlsx')
    xtest = pd.read_excel('./data/testiput.xlsx')
    ytrain = pd.read_excel('./data/trainlabel.xlsx')
    ytest = pd.read_excel('./data/testlabel.xlsx')
    # min-max scaler
    scaler = skpp.MinMaxScaler(feature_range=(0, 1))
    xntrain = scaler.fit_transform(xtrain)
    xntest = scaler.transform(xtest)
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)

    # train model
    method = method_list[0] # 0~10, 共11种方法
    if method == '不处理':
        print('选择不进行处理，程序运行中，请稍等')
        svm_org(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    elif method == 'BorderLine SMOTE 过采样':
        print('选择BorderLine SMOTE 过采样处理，程序运行中，请稍等')
        svm_bsmote(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    elif method == 'SMOTE过采样':
        print('选择SMOTE过采样处理，程序运行中，请稍等')
        svm_smote(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    elif method == '随机过采样':
        print('选择随机过采样处理，程序运行中，请稍等')
        svm_ros(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    elif method == '随机欠采样':
        print('选择随机欠采样处理，程序运行中，请稍等')
        svm_rus(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    elif method == 'NearMiss欠采样':
        print('选择NearMiss欠采样处理，程序运行中，请稍等')
        svm_nearmiss(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    elif method == 'RUSBoost':
        print('选择RUSBoost处理，程序运行中，请稍等')
        svm_RUSBoost(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    elif method == 'EasyEnsemble':
        print('选择EasyEnsemble处理，程序运行中，请稍等')
        svm_EasyEnsemble(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    elif method == 'BalanceCascade':
        print('选择BalanceCascade处理，程序运行中，请稍等')
        svm_BalanceCascade(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    elif method == '加权支持向量机':
        print('选择加权支持向量机处理，程序运行中，请稍等')
        svm_weight(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)            
    elif method == '焦点损失':
        print('选择焦点损失处理，程序运行中，请稍等')
        mlp_focal_loss(xntrain, ytrain, xntest, ytest, modelFolder, txtfolder)
    else:
        raise Exception('错误，未知的类不平衡处理方法')