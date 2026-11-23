"""Longevity model: which problems a stethoscope recording finds matter for survival, and how much improving them is worth.
→ web-independent result meta/longevity_model.json (read by the lite app's Longevity page and its calculator).

Data: NHANES 1999–2012 with NCHS linked mortality through 2019 (meta/nhanes_cohort.csv, pipeline/nhanes_prep.py), public domain.
The two problems the platform can find in a recording and NHANES also measured:
  * resting heart rate (stethoscope heart rate agrees with a research-grade ECG over a session — EPHNOGRAM test)
  * wheeze (NHANES RDQ070: wheezing or whistling in the chest in the past 12 months; self-report, not auscultation)

Rules fixed BEFORE the first result:
  * cohort: age 40–79 at the exam, eligible for mortality linkage, pulse 40–140 /min, complete covariates;
  * split by survey period, not at random: train 1999–2006, test 2007–2012 (≥ 7 years of follow-up for everyone tested);
  * outcome: all-cause death within 5 years (Cox models use the full follow-up, scored at 60 months);
  * models (training cycles only): A = Cox(age, sex) — the baseline; B = A + resting heart rate; C = B + wheeze + BMI + current and
    former smoking; G = gradient boosting on C's variables (machine-learning comparator). Deployed: C, unless G's test C-index
    beats C with a bootstrap CI above 0. Heart rate and wheeze "add prognostic information" only if C beats A with a CI above 0;
  * improvement scenarios use published effect sizes, not this data: exercise lowered resting heart rate by 3.3 beats/min against
    controls across 191 trials, and endurance training by up to 5.8 (Reimers et al., J Clin Med 2018, PMID 30513777).
    A predicted change is an association carried through the model, not a proven treatment effect; survey weights are not used.

  python3 pipeline/longevity_train.py
"""
import json, datetime
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from paths import META

SEED, H = 20261123, 60
V = ['age', 'female', 'hr10', 'wheeze', 'bmi', 'smoke_now', 'smoke_former']


def cohort():
    D = pd.read_csv(META / 'nhanes_cohort.csv')
    D = D[(D.ELIGSTAT == 1) & D.RIDAGEYR.between(40, 79) & D.BPXPLS.between(40, 140) & D.cycle.le(2011)].copy()
    D['age'] = D.RIDAGEYR; D['female'] = (D.RIAGENDR == 2).astype(int); D['hr10'] = D.BPXPLS / 10; D['bmi'] = D.BMXBMI
    D['wheeze'] = D.RDQ070.map({1: 1, 2: 0}); D['smoke_now'] = D.SMQ040.isin([1, 2]).astype(int)
    D['smoke_former'] = ((D.SMQ020 == 1) & (D.SMQ040 == 3)).astype(int); D.loc[D.SMQ020.isna(), ['smoke_now', 'smoke_former']] = np.nan
    D['T'] = D.PERMTH_EXM; D['E'] = (D.MORTSTAT == 1).astype(int)
    return D.dropna(subset=V + ['T'])


def risk_at(cph, X, t=H):
    s = cph.predict_survival_function(X, times=[t]).iloc[0].values; return 1 - s


def boot_c(df, a, b, B=1000):
    rng = np.random.default_rng(SEED); n = len(df); d = []
    for _ in range(B):
        i = rng.integers(0, n, n); x = df.iloc[i]
        d.append(concordance_index(x['T'], -x[a], x['E']) - concordance_index(x['T'], -x[b], x['E']))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    D = cohort(); tr = D[D.cycle <= 2005].copy(); te = D[D.cycle >= 2007].copy()
    sets = {'A': ['age', 'female'], 'B': ['age', 'female', 'hr10'], 'C': V}
    fits = {}
    for k, cols in sets.items():
        cph = CoxPHFitter(penalizer=0.0).fit(tr[cols + ['T', 'E']], 'T', 'E'); fits[k] = cph
        te['r' + k] = risk_at(cph, te[cols])
    # machine-learning comparator: 5-year death among those with 5 years of follow-up or death within 5 years
    lab = lambda d: d[(d['T'] >= H) | (d['E'] == 1)].assign(y=lambda x: ((x['E'] == 1) & (x['T'] < H)).astype(int))
    trl, tel = lab(tr), lab(te)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=SEED).fit(trl[V], trl['y'])
    te['rG'] = gb.predict_proba(te[V])[:, 1]
    C = {k: round(float(concordance_index(te['T'], -te['r' + k], te['E'])), 4) for k in 'ABCG'}
    tel = te.loc[tel.index]; auc = {k: round(float(roc_auc_score(lab(te)['y'], tel['r' + k])), 4) for k in 'ABCG'}
    dCA, dGC = boot_c(te, 'rC', 'rA'), boot_c(te, 'rG', 'rC')
    deployed = 'G' if dGC[0] > 0 else 'C'
    # calibration of the deployed Cox model at 5 years on the test cycles (deciles: predicted vs observed)
    from lifelines import KaplanMeierFitter
    te['dec'] = pd.qcut(te['rC'], 10, labels=False, duplicates='drop'); cal = []
    for q, g in te.groupby('dec'):
        km = KaplanMeierFitter().fit(g['T'], g['E']); cal.append({'pred': round(float(g['rC'].mean()), 4), 'obs': round(float(1 - km.survival_function_at_times(H).iloc[0]), 4), 'n': int(len(g))})
    c = fits['C']; summ = c.summary
    hr = {v: {'hr': round(float(np.exp(summ.loc[v, 'coef'])), 3), 'lo': round(float(np.exp(summ.loc[v, 'coef lower 95%'])), 3), 'hi': round(float(np.exp(summ.loc[v, 'coef upper 95%'])), 3)} for v in V}
    base = c.baseline_survival_; s0 = float(base[base.index <= H].iloc[-1, 0])
    ex = pd.DataFrame([{'age': 60, 'female': 0, 'hr10': 7.2, 'wheeze': 0, 'bmi': 27.0, 'smoke_now': 0, 'smoke_former': 0}])
    check = {'x': ex.iloc[0].to_dict(), 'risk': float(risk_at(c, ex[V])[0])}       # lifelines' own prediction; the page's formula must reproduce it
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED, 'horizonMonths': H,
           'source': 'NHANES 1999–2012 with NCHS Linked Mortality Files (follow-up to 31 Dec 2019), public domain',
           'cohort': {'train': int(len(tr)), 'test': int(len(te)), 'deathsTrain': int(tr.E.sum()), 'deathsTest': int(te.E.sum()), 'ages': [40, 79],
                      'trainCycles': '1999–2006', 'testCycles': '2007–2012', 'medianHr': float(D.BPXPLS.median()), 'wheezePct': round(100 * float(D.wheeze.mean()), 1)},
           'test': {'cIndex': C, 'auc5y': auc, 'deltaC_vs_ageSex': {'value': round(C['C'] - C['A'], 4), 'ci': dCA}, 'deltaC_gb_vs_cox': {'value': round(C['G'] - C['C'], 4), 'ci': dGC},
                    'addsInformation': bool(dCA[0] > 0), 'deployed': deployed, 'calibration': cal},
           'hazardRatios': hr,
           'cox': {'vars': V, 'coef': [float(c.params_[v]) for v in V], 'mean': [float(c._norm_mean[v]) for v in V], 's0': s0, 'check': check},
           'scenarios': [{'label': 'Exercise (all trials)', 'deltaHr': -3.3, 'ref': 'Reimers et al., J Clin Med 2018 · PMID 30513777'},
                         {'label': 'Endurance training (upper estimate)', 'deltaHr': -5.8, 'ref': 'Reimers et al., J Clin Med 2018 · PMID 30513777'}],
           'refs': [  # published Nov 2025 – Nov 2026 (PubMed-checked 2026-11-23), plus the meta-analysis the bpm effects come from
                    {'pmid': '42457225', 'doi': '10.1136/openhrt-2026-004279', 'cite': 'Chen et al. Resting heart rate and long-term mortality in young adults: a population-based cohort study. Open Heart 2026',
                     'why': 'per 10 bpm: all-cause mortality HR 1.26 (NHANES 1999–2004, age 20–49) and 1.17 (NHANES III) — the same size as this model'},
                    {'pmid': '42355572', 'doi': '10.3390/jcm15124405', 'cite': 'Cheon et al. Heart rate, electrocardiographic subclinical myocardial injury, and long-term mortality. J Clin Med 2026',
                     'why': 'resting tachycardia: all-cause mortality HR 3.58; silent myocardial injury explains only a small part of the link'},
                    {'pmid': '41483539', 'doi': '10.1016/j.hrtlng.2025.102713', 'cite': 'Yu et al. Comorbidity patterns of respiratory symptoms and diseases and their impact on all-cause and respiratory mortality. Heart Lung 2026',
                     'why': 'NHANES 2007–2011: the asthma–wheezing cluster was not linked with mortality after full adjustment (including lung function); obstruction and chronic cough were'},
                    {'pmid': '42973833', 'doi': '10.3389/fpubh.2026.1869982', 'cite': 'Wei et al. Recreational football as medicine? A meta-analysis of its effects and feasibility in patients with chronic diseases. Front Public Health 2026',
                     'why': '17 controlled trials: resting heart rate fell (SMD −1.01) along with blood pressure and fat mass'},
                    {'pmid': '42420366', 'doi': '10.1038/s41598-026-60452-7', 'cite': 'Mir et al. Efficacy of mental and physical training on blood pressure and resting heart rate among physically inactive young adults: a randomized controlled trial. Sci Rep 2026',
                     'why': '6 weeks of aerobic training lowered resting heart rate'},
                    {'pmid': '30513777', 'doi': '10.3390/jcm7120503', 'cite': 'Reimers AK et al. Effects of exercise on the resting heart rate: a systematic review and meta-analysis of interventional studies. J Clin Med 2018',
                     'why': 'source of the beats/min used in the scenarios (191 trials); no newer meta-analysis reports the effect in beats/min for adults in general'}]}
    (META / 'longevity_model.json').write_text(json.dumps(out, indent=1))
    print('cohort', out['cohort']); print('C-index', C, 'AUC5y', auc); print('ΔC (C−A)', out['test']['deltaC_vs_ageSex'], 'ΔC (GB−Cox)', out['test']['deltaC_gb_vs_cox'], 'deployed', deployed)
    print('HR', {k: v for k, v in hr.items()}); print('calibration', cal[:3], '…', cal[-2:])


if __name__ == '__main__':
    main()
