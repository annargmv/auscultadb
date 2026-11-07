"""Analysis 8: one aggregate score per child (method of Google's wearable hypertension study).
Per-recording features at expert-Normal sites are averaged per child -> regularised logistic regression, 5-fold CV.
Writes D['probe'] into web/data/discover.json (run after discover.py)."""
import json,glob,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from paths import SPR_FEATURES, WEB_DATA
R=[json.loads(l) for f in SPR_FEATURES for l in open(f)]
F=['cen','f50','f75','crkRate','fineRate','tonFrac']
kids={}
for r in R:
    if r['label']!='Normal' or r['rel'] not in('BioCAS2022','BioCAS2023','BioCAS2024'):continue
    if r['disease'] is None:continue
    k=kids.setdefault(r['pid'],{'age':r['age'],'rel':r['rel'],'ctrl':r['disease']=='Control Group','dx':r['disease'],'x':[]})
    k['x'].append([r[f] or 0 for f in F])
K=[k for k in kids.values() if k['x']]
X=np.array([np.mean(k['x'],0) for k in K]); age=np.array([[k['age']] for k in K])
rel=np.array([[k['rel']==r for r in('BioCAS2022','BioCAS2023')] for k in K],float)
y=np.array([0 if k['ctrl'] else 1 for k in K])
print('children',len(K),'sick',y.sum(),'controls',(1-y).sum())
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def auc(Z,name,reps=20):
    a=[]
    for s in range(reps):
        cv=StratifiedKFold(5,shuffle=True,random_state=s)
        m=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,class_weight='balanced',max_iter=2000))
        p=cross_val_predict(m,Z,y,cv=cv,method='predict_proba')[:,1]; a.append(roc_auc_score(y,p))
    print(f'{name:40s} AUC {np.mean(a):.3f} ± {np.std(a):.3f}'); return np.mean(a)
out={}
out['age']=auc(age,'age only')
out['release']=auc(rel,'recording batch only (confound check)')
out['agerel']=auc(np.hstack([age,rel]),'age + batch')
out['sound']=auc(X,'sound features (Normal sites only)')
out['all']=auc(np.hstack([X,age,rel]),'sound + age + batch')
# within-release
for rl in('BioCAS2022','BioCAS2024'):
    idx=np.array([k['rel']==rl for k in K]); yy=y[idx]
    a=[]
    for s in range(20):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,class_weight='balanced',max_iter=2000))
        p=cross_val_predict(m,np.hstack([X,age])[idx],yy,cv=StratifiedKFold(5,shuffle=True,random_state=s),method='predict_proba')[:,1];a.append(roc_auc_score(yy,p))
    print(rl,'n',idx.sum(),'controls',(1-yy).sum(),'AUC sound+age',round(np.mean(a),3))
    out['within_'+rl]=[int(idx.sum()),int((1-yy).sum()),float(np.mean(a))]
m=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,class_weight='balanced',max_iter=2000)).fit(np.hstack([X,age]),y)
print(dict(zip(F+['age'],np.round(m[-1].coef_[0],2))))
out['coef']=dict(zip(F+['age'],[float(c) for c in m[-1].coef_[0]])); out['n']=[len(K),int(y.sum()),int((1-y).sum())]
D=json.load(open(WEB_DATA/'discover.json')); D['probe']=out; json.dump(D,open(WEB_DATA/'discover.json','w'),separators=(',',':'))
