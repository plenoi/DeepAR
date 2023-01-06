import numpy as np
import pandas as pd

#!pip install jpype1
#!pip install torch==1.9.1
#!pip install scikit-learn==0.24.1
#!pip install xgboost==0.90

file = open('PLS.py','w')
file.write('import numpy as np'+"\n")
file.write('from sklearn.cross_decomposition import PLSRegression'+"\n")
file.write('from sklearn.base import BaseEstimator, ClassifierMixin'+"\n")
file.write('class PLS(BaseEstimator, ClassifierMixin):'+"\n")
file.write('    def __init__(self):'+"\n")
file.write('        self.clf = PLSRegression(n_components=2)'+"\n")
file.write('    def fit(self, X, y):'+"\n")
file.write('        self.clf.fit(X,y)'+"\n")
file.write('        return self'+"\n")
file.write('    def predict(self, X):'+"\n")
file.write('        pr = [np.round(min(max(np.round(item[0]),0.000001),0.999999)) for item in self.clf.predict(X)]'+"\n")
file.write('        return np.array(pr)'+"\n")
file.write('    def predict_proba(self, X):'+"\n")
file.write('        p_all = []'+"\n")
file.write('        ptmp = np.array([min(max(item[0],0.000001),0.999999) for item in self.clf.predict(X)],dtype=float)'+"\n")
file.write('        p_all.append(1-ptmp)'+"\n")
file.write('        p_all.append(ptmp)'+"\n")
file.write('        return np.transpose(np.array(p_all))'+"\n")
file.close()

from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage
if not isJVMStarted():
    cdk_path = '../input/nuclear-smile/cdk-2.7.1.jar'
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % cdk_path)
    cdk =  JPackage('org').openscience.cdk
    
def featsmi(fp_type, smis, size=1024, depth=6):
    fg = {
            "AP2D" : cdk.fingerprint.AtomPairs2DFingerprinter(),
            "CKD":cdk.fingerprint.Fingerprinter(size, depth),
            "CKDExt":cdk.fingerprint.ExtendedFingerprinter(size, depth),
            "CKDGraph":cdk.fingerprint.GraphOnlyFingerprinter(size, depth),
            "MACCS":cdk.fingerprint.MACCSFingerprinter(),
            "PubChem":cdk.fingerprint.PubchemFingerprinter(cdk.silent.SilentChemObjectBuilder.getInstance()),
            "Estate":cdk.fingerprint.EStateFingerprinter(),
            "KR":cdk.fingerprint.KlekotaRothFingerprinter(),
            "FP4" : cdk.fingerprint.SubstructureFingerprinter(),
            "FP4C" : cdk.fingerprint.SubstructureFingerprinter(),
            "Circle" : cdk.fingerprint.CircularFingerprinter(),
            "Hybrid" : cdk.fingerprint.HybridizationFingerprinter(),
         }
    sp = cdk.smiles.SmilesParser(cdk.DefaultChemObjectBuilder.getInstance())
    for i,smi in enumerate(smis):
        mol = sp.parseSmiles(smi)
        if fp_type == "FP4C":
            fingerprinter = fg[fp_type]
            nbit = fingerprinter.getSize()
            fp = fingerprinter.getCountFingerprint(mol)
            feat = np.array([int(fp.getCount(i)) for i in range(nbit)])           
        else:
            fingerprinter = fg[fp_type]
            nbit = fingerprinter.getSize()
            fp = fingerprinter.getFingerprint(mol)
            feat = np.array([int(fp.get(i)) for i in range(nbit)])
        if i == 0:
            featx = feat.reshape(1,-1)
        else:
            featx = np.vstack((featx, feat.reshape(1,-1)))
    return featx
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(CNN_NLP, self).__init__()
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=1,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        x_conv_list = [F.relu(conv1d(input_ids)) for conv1d in self.conv1d_list]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        logits = self.fc(self.dropout(x_fc))
        return logits
 
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,SequentialSampler)
import random

loss_fn = nn.CrossEntropyLoss()
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def data_loader(train_inputs, val_inputs, train_labels, val_labels,batch_size=200):
    g = torch.Generator()
    g.manual_seed(0)
    train_inputs = torch.from_numpy(np.array(train_inputs)).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_inputs = torch.from_numpy(np.array(val_inputs)).float()
    val_labels = torch.from_numpy(val_labels).long()
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=1, worker_init_fn=seed_worker,generator=g,)
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, num_workers=1, worker_init_fn=seed_worker,generator=g,)
    return train_dataloader, val_dataloader

df = pd.read_csv('./input/smiles.csv', names=['Smiles'], header=None)
data = df['Smiles'].values

import joblib
yt = np.zeros(len(data)) # For prediction only, if u don't know groud-truth
stackar = joblib.load("./model/allmodelnuclear.sav")
scaler = joblib.load("./model/FP4C_Scaler.sav") 
fname = ['AP2D','CKD','CKDExt','CKDGraph','MACCS','PubChem','Estate','KR','FP4','FP4C','Circle','Hybrid']
feat_AP2D = featsmi("AP2D",data)
feat_CKD = featsmi("CKD",data)
feat_CKDExt = featsmi("CKDExt",data)
feat_CKDGraph = featsmi("CKDGraph",data)
feat_MACCS = featsmi("MACCS",data)
feat_PubChem = featsmi("PubChem",data)
feat_Estate = featsmi("Estate",data)
feat_KR = featsmi("KR",data)
feat_FP4 = featsmi("FP4",data)
feat_FP4C = featsmi("FP4C",data)
feat_Circle = featsmi("Circle",data)
feat_Hybrid = featsmi("Hybrid",data)
feat_FP4C =  scaler.transform(feat_FP4C)

k = 0; kk = 0
for i in range(len(stackar)):
    pr = stackar[i].predict_proba(eval("feat_"+fname[kk]))[:,0]
    Xst = pr.reshape(-1,1) if i==0 else np.hstack((Xst,pr.reshape(-1,1)))
    if k == 9:
        kk = kk + 1; k = 0
    else:
        k = k + 1
        
device = 'cpu'
model = CNN_NLP()
model.to(device)
model.load_state_dict(torch.load("./model/AR.pt"))
Xst = Xst.reshape(Xst.shape[0],1,Xst.shape[1])
_, test_dl  = data_loader(Xst, Xst, yt, yt, batch_size=len(yt))
batch = next(iter(test_dl))
ts, _ = batch
model.eval()
prob =  model(ts)
pr = np.array([1-F.softmax(item,0)[0].item() for item in prob])  

label = ['Positive', 'Negative'] 
file = open("./output/predict_result.csv","w")
for i, head, in enumerate(data):
    file.write(head+","+label[int(pr[i]+0.5)]+","+str(1-pr[i])+"\n")
file.close()