"""Offline-only reproduction. Uses local released files; never fetches data."""
import os
import sys
import sysconfig
from pathlib import Path
# On macOS, reuse the OpenMP library shipped by sklearn instead of installing
# a global Homebrew package. Must be configured before the interpreter loads it.
if sys.platform=='darwin' and not os.environ.get('T1_OPENMP_READY'):
    omp=Path(sysconfig.get_paths()['purelib'])/'sklearn/.dylibs'
    if (omp/'libomp.dylib').exists():
        os.environ['DYLD_LIBRARY_PATH']=str(omp)+(':'+os.environ['DYLD_LIBRARY_PATH'] if os.environ.get('DYLD_LIBRARY_PATH') else '')
        os.environ['T1_OPENMP_READY']='1'
        os.execv(sys.executable,[sys.executable,*sys.argv])
import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import numpy as np
import torch
from src.data import load_sales,make_samples,audit,tensor_dataset
from src.experiment import train,train_gbm,base_policies,neural_policy,gbm_policy,simulate,select_channel,predict
from src.inventory import normal_order,gamma_order


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def aggregate(rows):
    result=[]
    for policy in dict.fromkeys(r['policy'] for r in rows):
        values=[r for r in rows if r['policy']==policy]
        result.append(dict(policy=policy,n_skus=len(values),**{k:float(np.mean([r[k] for r in values]))
                      for k in ['holding_cost','stockout_cost','total_cost']}))
    return result


def release_diagnostics(data_dir,output,c):
    with (data_dir/'df_test.csv').open() as handle:rows=list(csv.DictReader(handle))
    targets=np.array([float(r['OPT_pred']) for r in rows]);diagnostics=[]
    values={name:np.array([float(r[name]) for r in rows]) for name in ['E2E_RNN_pred','gbm_pred']}
    values['Normal_eq7']=np.array([normal_order(float(r['demand_mean']),float(r['demand_std']),float(r['review_period']),
       float(r['vendor_vlt_mean']),float(r['vendor_vlt_std']),float(r['initial_stock']),c['holding_cost'],c['stockout_cost']) for r in rows])
    values['Gamma_eq8']=np.array([gamma_order(float(r['demand_mean']),float(r['demand_std']),float(r['review_period']),
       float(r['vendor_vlt_mean']),float(r['initial_stock']),c['holding_cost'],c['stockout_cost']) for r in rows])
    for name,prediction in values.items():
        errors=prediction-targets
        diagnostics.append(dict(model=name,n=len(rows),order_mae=float(np.abs(errors).mean()),
           order_rmse=float(np.sqrt((errors**2).mean())),inventory_cost_available=False,
           source='author_prediction' if name in ['E2E_RNN_pred','gbm_pred'] else 'equation_implementation'))
    write_csv(output/'released_3000_diagnostics.csv',diagnostics)


def run(args):
    c=json.loads(args.config.read_text());output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.epochs is not None:c['epochs']=args.epochs
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    data=args.data_dir.resolve();sales,ids=load_sales(data);audit(data,output/'data_manifest.json')
    # Confirm the fixed lead fixture is transparently derived from the sole VLT
    # example. It is not linked to these five anonymized sales identifiers.
    with (data/'E2E-model-data/Data/vlt_2019_0305.csv').open() as handle:vlt=list(csv.DictReader(handle))
    inferred=int(np.ceil(float(vlt[0]['vlt_actual'])))
    if inferred!=c['lead_time']:raise ValueError('Config L differs from ceil of the released VLT example')
    release_diagnostics(data,output,c)
    (output/'checkpoints').mkdir(exist_ok=True)
    cache={};histories=[];experiment_records=[];all_traces=[];verification=[]
    def fit(config,mode='e2e'):
        key=json.dumps([config,mode],sort_keys=True)
        if key in cache:return cache[key]
        samples=make_samples(sales,config)
        model,logs,meta=train(samples,config,len(sales),mode)
        tag=f'{mode}_{hashlib.sha256(key.encode()).hexdigest()[:10]}'
        for row in logs:histories.append(dict(run=tag,mode=mode,**row))
        path=output/'checkpoints'/f'{tag}.pt'
        torch.save(dict(state_dict=model.state_dict(),config=config,sku_count=len(sales),mode=mode),path)
        # Every saved model must regenerate the same predictions after reload.
        from src.network import InventoryNetwork
        saved=torch.load(path,map_location='cpu',weights_only=True)
        loaded=InventoryNetwork(saved['config'],saved['sku_count']);loaded.load_state_dict(saved['state_dict']);loaded.eval()
        batch=tensor_dataset(samples['test'][:3],config).tensors
        with torch.no_grad():
            for actual,expected in zip(loaded(*batch[:5]),model(*batch[:5])):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        verification.append(dict(run=tag,reload_exact=True))
        result=(model,meta,tag);cache[key]=result
        print(f'{tag}: {meta["n_train"]} train samples, best epoch {meta["best_epoch"]}, {meta["training_seconds"]:.1f}s',flush=True)
        return result
    e2e,meta,tag=fit(c)
    bm1,_,bm1tag=fit(c,'bm1');bm2,_,bm2tag=fit(c,'bm2')
    channel1,candidates1=select_channel(bm1,sales,c,'bm1');channel2,candidates2=select_channel(bm2,sales,c,'bm2')
    samples=make_samples(sales,c);gbm=train_gbm(samples,c)
    gbm.booster_.save_model(str(output/'checkpoints/gbm.txt'))
    policies=base_policies(c)+[('E2E_RNN',neural_policy(e2e,c,len(sales),'e2e')),
        ('E2E_GBM',gbm_policy(gbm,c,len(sales))),('BM1',neural_policy(bm1,c,len(sales),'bm1',channel1)),
        ('BM2',neural_policy(bm2,c,len(sales),'bm2',channel2))]
    rows,traces=simulate(sales,c,policies);all_traces=traces
    write_csv(output/'figure_2b_costs.csv',aggregate(rows));write_csv(output/'figure_2b_per_sku.csv',rows)
    write_csv(output/'inventory_trajectories.csv',traces)
    # Figure 2(a): forecast from first evaluation day, first three fixed SKUs.
    forecast_rows=[];start=sales.shape[1]-c['evaluation_days'];horizon=c['lead_time']+c['review_period']
    for sku in range(min(3,len(sales))):
        history=sales[sku,start-c['history']:start]
        _,forecast,_,_=predict(e2e,history,sku,len(sales),history.mean()*horizon,c)
        for day in range(horizon):
            forecast_rows.append(dict(sku_index=sku,horizon_day=day,observed=sales[sku,start+day],mean=forecast[day,0],
               q10=forecast[day,1+c['quantiles'].index(.1)],q50=forecast[day,1+c['quantiles'].index(.5)],q90=forecast[day,1+c['quantiles'].index(.9)]))
    write_csv(output/'figure_2a_forecasts.csv',forecast_rows)
    groups=[('B2_layers','fusion_layers',[[100],[100,100],[100,100,100],[100,100,100,100]]),
            ('B3_hidden','hidden',[30,40,50,60]),('B4_learning_rate','learning_rate',[.0001,.001,.01]),
            ('B5_training_fraction','train_fraction',[.2,.4,.6,.8,1.]),
            ('B2_cost_ratio','stockout_cost',[1.,3.,9.,19.]),('B6_seed','seed',[7,17])]
    if args.suite=='main':groups=[]
    seed_weights=[dict(seed=c['seed'],weights=e2e.vlt[0].weight.detach().numpy().tolist())]
    for group,parameter,values in groups:
        for value in values:
            config=copy.deepcopy(c);config[parameter]=value
            model,run_meta,run_tag=fit(config)
            per_sku,trace=simulate(sales,config,base_policies(config)+[('E2E_RNN',neural_policy(model,config,len(sales),'e2e'))])
            for score in aggregate(per_sku):
                experiment_records.append(dict(experiment=group,parameter=parameter,value=json.dumps(value),
                    run=run_tag,**score,**run_meta))
            if group=='B6_seed' and value!=c['seed']:
                seed_weights.append(dict(seed=value,weights=model.vlt[0].weight.detach().numpy().tolist()))
    if experiment_records:write_csv(output/'appendix_sensitivity.csv',experiment_records)
    write_csv(output/'training_history.csv',histories)
    write_csv(output/'sample_splits.csv',[dict(part=part,sku=s['sku'],start=s['t'],end_exclusive=s['end'])
                                        for part,values in samples.items() for s in values])
    (output/'vlt_weights.json').write_text(json.dumps(seed_weights,indent=2)+'\n')
    # Verify every policy's total cost is bounded by the deterministic OPT cost.
    oracle={r['sku_index']:r['total_cost'] for r in rows if r['policy']=='OPT'}
    for row in rows:
        if row['total_cost']<oracle[row['sku_index']]-1e-5:raise AssertionError('Policy beat perfect-information OPT')
    (output/'verification.json').write_text(json.dumps(dict(checkpoints=verification,oracle_bounds_pass=True),indent=2)+'\n')
    (output/'run_manifest.json').write_text(json.dumps(dict(config=c,suite=args.suite,sku_count=len(sales),sales_days=sales.shape[1],
        samples={k:len(v) for k,v in samples.items()},python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('.').glob('src/*.py'))},
        bm1_validation_selected_channel=channel1,bm2_validation_selected_channel=channel2,
        bm1_validation_candidates=candidates1,bm2_validation_candidates=candidates2,
        architecture_run=tag,bm1_run=bm1tag,bm2_run=bm2tag,
        limitations=['fixed lead from one unlinked example','constructed stock states','five sales series only',
                     'full original 132-dimensional features missing','Section 5 intentionally excluded']),indent=2)+'\n')
    from src.report import render
    render(output)
    print(f'Completed {len(cache)} neural trainings; results: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/paper.json'))
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--suite',choices=['main','all'],default='all')
    parser.add_argument('--epochs',type=int,help='Explicit diagnostic override; recorded in manifest')
    args=parser.parse_args()
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],check=True)
    run(args)
