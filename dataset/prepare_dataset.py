import python as pd
import numpy as np

df = pd.read_csv('./input/AR.csv')
smiles = df['Smiles'].values
y_all = df['Activity'].values
y_all[y_all=='active'] = 0
y_all[y_all=='inactive'] = 1
all_pos_idx = np.where(y_all==0)[0]
all_neg_idx = np.where(y_all==1)[0]
numPos = sum(y_all==0)
numNeg = sum(y_all==1)
rng = np.random.RandomState(0)
pos_idx_tr = list(rng.choice(all_pos_idx, int(numPos*0.8), replace=False))
neg_idx_tr = list(rng.choice(all_neg_idx, int(numNeg*0.8), replace=False))
pos_idx_ts = list(set(all_pos_idx) - set(pos_idx_tr))
neg_idx_ts = list(set(all_neg_idx) - set(neg_idx_tr))

smiles_tr = smiles[pos_idx_tr + neg_idx_tr]  # Train
smiles_ts = smiles[pos_idx_ts + neg_idx_ts]  # Test
y = np.hstack((np.zeros(len(pos_idx_tr)), np.ones(len(neg_idx_tr))))  # Train Label
yt = np.hstack((np.zeros(len(pos_idx_ts)), np.ones(len(neg_idx_ts)))) # Test Label