"""Only local released data; no external downloads and no invented real records."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from .inventory import label


def load_sales(data_dir):
    path=Path(data_dir)/'E2E-model-data/Data/rdc_sales_1320.csv'
    with path.open() as handle:
        reader=csv.DictReader(handle);columns=reader.fieldnames[1:];rows=list(reader)
    if list(map(int,columns))!=list(range(int(columns[0]),int(columns[0])+len(columns))):
        raise ValueError('Expected consecutive day indices')
    sales=np.array([[float(row[k]) for k in columns] for row in rows])
    if not np.isfinite(sales).all() or (sales<0).any():raise ValueError('Invalid sales')
    return sales,[r['row'] for r in rows]


def features(history,sku,sku_count,stock,config):
    history=np.asarray(history,float)
    scale=max(float(history.mean()),1.)
    return [history[:,None]/scale,np.eye(sku_count)[sku],np.array([config['lead_time']]),
            np.array([config['review_period']]),np.array([stock/scale])],scale


def make_samples(sales,config):
    c=config; n=sales.shape[1];horizon=c['lead_time']+c['review_period']
    cuts=[c['history'],int(n*c['train_end_fraction']),int(n*c['validation_end_fraction']),n]
    sets={}
    for part,start,end in zip(['train','validation','test'],cuts[:-1],cuts[1:]):
        times=list(range(start,end-horizon+1,c.get('sampling_stride',7)))
        if part=='train':times=times[:max(1,math.ceil(len(times)*c['train_fraction']))]
        values=[]
        for sku,series in enumerate(sales):
            for t in times:
                history=series[t-c['history']:t];future=series[t:t+horizon]
                for cover in c['inventory_cover_fractions']:
                    stock=history.mean()*horizon*cover
                    inputs,scale=features(history,sku,len(sales),stock,c)
                    target=label(future[c['lead_time']:],stock-future[:c['lead_time']].sum(),c['holding_cost'],c['stockout_cost'])
                    values.append(dict(inputs=inputs,scale=scale,future=future,target=target,
                                       stock=stock,sku=sku,t=t,end=t+horizon,part=part))
        if not values:raise ValueError(f'No usable {part} samples')
        sets[part]=values
    return sets


def tensor_dataset(samples,config):
    f=lambda values:torch.tensor(np.asarray(values),dtype=torch.float32)
    return TensorDataset(*[f([s['inputs'][i] for s in samples]) for i in range(5)],
                         f([s['target']/s['scale'] for s in samples]),
                         f([s['future']/s['scale'] for s in samples]),
                         f([config['lead_time'] for _ in samples]))


def gbm_features(sample):
    h=sample['inputs'][0].ravel()
    return np.r_[h.mean(),h.std(),h[-7:].mean(),h[-28:].mean(),
                 h[-7:].mean()-h[-14:-7].mean(),h[-1]-h[-2],
                 sample['inputs'][1],sample['inputs'][2],sample['inputs'][3],sample['inputs'][4]]


def audit(data_dir,output):
    files=[]
    for path in sorted(Path(data_dir).rglob('*.csv')):
        with path.open() as handle:
            reader=csv.DictReader(handle);rows=list(reader)
        files.append(dict(file=str(path.relative_to(data_dir)),rows=len(rows),columns=reader.fieldnames,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    Path(output).write_text(json.dumps(files,indent=2)+'\n')
    return files
