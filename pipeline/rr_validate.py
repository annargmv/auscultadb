"""Breathing rate against the ICBHI experts' cycles → meta/rr_validation.json.

Reference: 60 / median length of the expert-annotated respiratory cycles of each recording (≥ 3 cycles). Index tests:
  * the page's shipped estimator (recordingFeatures → rr, from analyzeSignal), as stored in meta/features/icbhi.jsonl;
  * a candidate: autocorrelation of the log band envelope (125–1000 Hz) of the page's cnnSpec, lag 1–8 s; band and
    smoothing chosen on the ICBHI TRAINING patients only (icbhi_models.json testPids held out).
Rule fixed before scoring: the candidate replaces the shipped estimator only if its MAE reduction on held-out patients has a
95 % patient-bootstrap CI above 0. Research licence: only numbers leave build/.

  python3 pipeline/rr_validate.py
"""
import json,numpy as np,sys
from paths import ROOT, META, FEATURES, WEB_DATA
from scipy import stats
D=str(ROOT/'build'/'icbhi_cnn')+'/'
idx=[json.loads(l) for l in open(D+'index.jsonl')]; F=np.fromfile(D+'spec.f32',dtype='<f4').reshape(-1,64)
IM=json.load(open(str(WEB_DATA/'icbhi_models.json'))); test=set(IM['testPids'])
I=json.load(open(str(META/'icbhi_index.json'))); age={p['pid']:p['age'] for p in I['patients']}
feat={json.loads(l)['name']:json.loads(l)['rf'].get('rr') for l in open(str(FEATURES/'icbhi.jsonl'))}
def est(B,lo_bin,hi_bin,smooth,lagmin,lagmax):
    e=np.log10(np.sum(10**B[:,lo_bin:hi_bin],1)+1e-10); k=max(1,int(smooth/0.01)); e=np.convolve(e,np.ones(k)/k,'same'); e=e-e.mean()
    n=len(e); L0,L1=int(lagmin/0.01),min(int(lagmax/0.01),n//2)
    if L1<=L0+2: return None
    ac=np.array([np.dot(e[:n-L],e[L:])/(n-L) for L in range(L0,L1)])
    # first strong local maximum (>= 0.6 of the global max) avoids picking a 2-breath multiple
    g=ac.max(); 
    for i in range(1,len(ac)-1):
        if ac[i]>=ac[i-1] and ac[i]>=ac[i+1] and ac[i]>=0.6*g: return 60/((L0+i)*0.01)
    return 60/((L0+int(np.argmax(ac)))*0.01)
rows=[]
for r in idx:
    c=r['cycles']
    if len(c)<3 or r['nF']<600: continue
    rows.append((r['pid'],r['name'],60/np.median([b-a for a,b,_,_ in c]),F[r['offset']:r['offset']+r['nF']],feat.get(r['name'])))
tr=[x for x in rows if x[0] not in test]; te=[x for x in rows if x[0] in test]
best=None
for lo,hi in ((2,16),(4,32),(4,16),(2,32),(8,32)):
  for sm in (0.2,0.4,0.6):
    for thr in (0.6,):
      p=[est(x[3],lo,hi,sm,1.0,8.0) for x in tr]; t=np.array([x[2] for x in tr]); p=np.array([np.nan if v is None else v for v in p]); ok=~np.isnan(p)
      mae=np.mean(np.abs(p[ok]-t[ok])); 
      if best is None or mae<best[0]: best=(mae,lo,hi,sm)
print('best on TRAIN',best)
_,lo,hi,sm=best
t=np.array([x[2] for x in te]); p=np.array([np.nan if v is None else v for v in [est(x[3],lo,hi,sm,1.0,8.0) for x in te]]); o=np.array([np.nan if x[4] is None else x[4] for x in te])
ok=~np.isnan(p)&~np.isnan(o)
for name,v in (('new',p),('page',o)):
    d=v[ok]-t[ok]; print(name,'held-out n',ok.sum(),'MAE',np.abs(d).mean().round(2),'bias',d.mean().round(2),'LoA ±',(1.96*d.std()).round(1),'rho',stats.spearmanr(v[ok],t[ok])[0].round(3),'within 3/min',(np.abs(d)<=3).mean().round(3))
pid=np.array([x[0] for x in te])[ok]; ae_new=np.abs(p[ok]-t[ok]); ae_old=np.abs(o[ok]-t[ok]); up=np.unique(pid); ix={u:np.where(pid==u)[0] for u in up}; rng=np.random.default_rng(20261118); dd=[]
for _ in range(2000):
    i=np.concatenate([ix[u] for u in rng.choice(up,len(up))]); dd.append(ae_old[i].mean()-ae_new[i].mean())
print('MAE reduction',(ae_old.mean()-ae_new.mean()).round(2),'CI',np.percentile(dd,[2.5,97.5]).round(2),'patients',len(up))
ag=np.array([age[x[0]] or 0 for x in te])[ok]
for g,k in (('children',ag<18),('adults',ag>=18)):
    print(g,k.sum(),'new MAE',ae_new[k].mean().round(2),'page MAE',ae_old[k].mean().round(2))

def summ(v,mask):
    d=v[mask]-t[mask]; return {'n':int(mask.sum()),'mae':round(float(np.abs(d).mean()),2),'bias':round(float(d.mean()),2),'loa':[round(float(d.mean()-1.96*d.std()),1),round(float(d.mean()+1.96*d.std()),1)],
            'rho':round(float(stats.spearmanr(v[mask],t[mask])[0]),3),'within3':round(100*float((np.abs(d)<=3).mean()),1)}
out={'what':'Breathing rate vs ICBHI expert cycles (60 / median cycle length), held-out patients','patients':int(len(up)),'recordings':int(ok.sum()),
     'reference':{'median':round(float(np.median(t[ok])),1)},'shipped':summ(o,ok),'candidate':{**summ(p,ok),'band':[lo,hi],'smoothS':sm},
     'reduction':{'mae':round(float(ae_old.mean()-ae_new.mean()),2),'ci':[round(float(x),2) for x in np.percentile(dd,[2.5,97.5])]},
     'byGroup':[{'group':g,'recordings':int(k.sum()),'maeShipped':round(float(ae_old[k].mean()),2),'maeCandidate':round(float(ae_new[k].mean()),2)} for g,k in (('children',ag<18),('adults',ag>=18))]}
out['adopt']=bool(out['reduction']['ci'][0]>0)
json.dump(out,open(META/'rr_validation.json','w'),indent=1); print('->',META/'rr_validation.json','adopt candidate:',out['adopt'])
