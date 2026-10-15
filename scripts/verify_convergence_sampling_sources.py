#!/usr/bin/env python3
"""Verify/export frozen convergence and sampling summaries. No B22 recalculation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('config/convergence_sampling.json'))
    args=p.parse_args(); cp=args.config.resolve(strict=True)
    config=json.loads(cp.read_text()); root=cp.parent.parent
    source=root/config['frozen_sources']
    summary=pd.read_csv(source/'Table_S7A.csv').dropna(subset=['k']).sort_values('k')
    np.testing.assert_array_equal(summary.k,[1,2,3,4,5])
    np.testing.assert_allclose(summary.median_absolute_deviation,[121.073486,72.685436,48.456957,30.268371,0],atol=1e-6)
    columns=['fraction_within_5pct','fraction_within_10pct','fraction_within_20pct']
    np.testing.assert_allclose(summary[columns],[[.244275,.441439,.710578],[.368593,.634351,.883751],
        [.514177,.794111,.963359],[.710578,.928244,.994766],[1,1,1]],atol=1e-6)
    time=pd.read_csv(source/'Table_S7B.csv').dropna(subset=['pair_id'])
    wide=time.pivot(index=['pair_id','regime'],columns='trajectory_fraction',values='mean')
    if len(wide)!=15: raise ValueError('Expected original 15 trajectory-convergence pairs.')
    np.testing.assert_array_equal(wide.columns,[.2,.4,.6,.8,1.])
    change=100*abs(wide[.8]-wide[1.])/abs(wide[1.])
    if sum(change<5)!=13 or sum(change<10)!=15: raise ValueError('Trajectory threshold mismatch.')
    medians=change.groupby(level='regime').median()
    for regime,value in [('weak',1.44751095069),('intermediate',1.08593760285),('strong',3.25665442705)]:
        np.testing.assert_allclose(medians[regime],value,atol=1e-8)
    stages=pd.read_csv(source/'Table_S6E.csv').dropna(subset=['stage_number'])
    np.testing.assert_array_equal(stages.cumulative_pair_count,[498,678,843,917])
    train=pd.read_csv(source/'Table_S6C.csv'); held=pd.read_csv(source/'Table_S6D.csv')
    if len(train)!=917 or train.pair_id.nunique()!=917 or len(held)!=1794: raise ValueError('Membership count mismatch.')
    conv=root/config['convergence_results']; sampling=root/config['sampling_results']
    conv.mkdir(parents=True,exist_ok=True); sampling.mkdir(parents=True,exist_ok=True)
    summary[['k','n_combinations','median_absolute_deviation','absolute_deviation_p05','absolute_deviation_p95']+columns].to_csv(conv/'replicate_count_convergence_summary.csv',index=False)
    change.rename('relative_change_80_to_100_pct').to_csv(conv/'trajectory_final_change.csv')
    stages[[c for c in ['stage_number','stage','cumulative_pair_count','attractive_pair_count',
        'log10_negB22_median','KS_log10_vs_final917','Wasserstein_log10_vs_final917'] if c in stages]].to_csv(sampling/'B22_distribution_stability_by_stage.csv',index=False)
    report=dict(training_pairs=917,heldout_pairs=1794,trajectory_pairs=15,change_below_5pct=13,
        change_below_10pct=15,median_change_pct=medians.to_dict(),B22_recomputed=False,simulations_run=False)
    (conv/'verification_report.json').write_text(json.dumps(report,indent=2))
    print(summary[['k','median_absolute_deviation']+columns].to_string(index=False))
    print(json.dumps(report,indent=2)); print('Results:',conv,sampling)

if __name__=='__main__': main()
