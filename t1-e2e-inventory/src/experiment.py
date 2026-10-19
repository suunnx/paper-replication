"""Train models and evaluate continuous 30-day inventory trajectories."""
import copy
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import tensor_dataset,features,gbm_features
from .network import InventoryNetwork,loss
from .inventory import label,normal_order,gamma_order,bm1_order,bm2_order,step


def train(samples,config,sku_count,mode='e2e'):
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=InventoryNetwork(config,sku_count)
    optimizer=torch.optim.SGD(model.parameters(),lr=config['learning_rate'],momentum=config['momentum'])
    train_data=tensor_dataset(samples['train'],config);val=tensor_dataset(samples['validation'],config).tensors
    loader=DataLoader(train_data,batch_size=config['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');best_state=None;stale=0;iteration=0;logs=[];start=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        model.train();total=0.
        for batch in loader:
            iteration+=1
            for group in optimizer.param_groups:group['lr']=config['learning_rate']/(1+config['learning_rate_decay']*iteration)
            optimizer.zero_grad();value=loss(model(*batch[:5]),*batch[5:],config,mode)
            if not torch.isfinite(value):raise ValueError(f'Nonfinite loss: {mode}, seed={seed}')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            total+=value.item()*len(batch[0])
        model.eval()
        with torch.no_grad():validation=loss(model(*val[:5]),*val[5:],config,mode).item()
        logs.append(dict(epoch=epoch,train_loss=total/len(train_data),validation_loss=validation))
        if validation<best-1e-7:
            best=validation;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=config['patience']:break
    model.load_state_dict(best_state);model.eval()
    return model,logs,dict(best_epoch=best_epoch,validation_loss=best,training_seconds=time.perf_counter()-start,
                           n_train=len(train_data),n_validation=len(val[0]))


def predict(model,history,sku,sku_count,stock,config):
    inputs,scale=features(history,sku,sku_count,stock,config)
    tensors=[torch.tensor(x[None],dtype=torch.float32) for x in inputs]
    with torch.no_grad():order,daily,lead,total=model(*tensors)
    return order.item()*scale,daily[0].numpy()*scale,lead.item(),total[0].numpy()*scale


def train_gbm(samples,config):
    from lightgbm import LGBMRegressor,early_stopping,log_evaluation
    x=np.array([gbm_features(s) for s in samples['train']]);y=np.array([s['target']/s['scale'] for s in samples['train']])
    vx=np.array([gbm_features(s) for s in samples['validation']]);vy=np.array([s['target']/s['scale'] for s in samples['validation']])
    model=LGBMRegressor(n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=10,
                       random_state=config['seed'],n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
    model.fit(x,y,eval_set=[(vx,vy)],callbacks=[early_stopping(20,verbose=False),log_evaluation(0)])
    return model


def simulate(sales,config,policies,*,start=None,days=None):
    """Continuous per-SKU state; scheduled arrivals before demand, backorders.

    L<R in the executable fixture, so no open order at the next review. General
    random-VLT/order-crossover empirical reconstruction is NOT claimed.
    A policy is (name, function(history, sku, stock, remaining_days)).
    """
    c=config;days=days or c['evaluation_days'];start=len(sales[0])-days if start is None else start
    if c['lead_time']>=c['review_period']:raise ValueError('Fixture requires L<R; pending-order features are unavailable')
    if start<c['history'] or start+days>sales.shape[1]:raise ValueError('Invalid evaluation window')
    rows=[];traces=[]
    for name,policy in policies:
        for sku,series in enumerate(sales):
            initial=series[start-c['history']:start].mean()*(c['lead_time']+c['review_period'])
            stock=float(initial);pipeline={};hold=short=0.
            for day in range(days):
                t=start+day;arrivals=pipeline.pop(day,0.);stock+=arrivals;order=0.
                if day%c['review_period']==0 and day+c['lead_time']<days:
                    if name=='OPT':
                        arrival_stock=stock-series[t:t+c['lead_time']].sum()
                        stop=min(days,day+c['lead_time']+c['review_period'])
                        order=label(series[t+c['lead_time']:start+stop],arrival_stock,c['holding_cost'],c['stockout_cost'])
                    else:order=max(float(policy(series[t-c['history']:t],sku,stock,days-day)),0.)
                    if not np.isfinite(order):raise ValueError('Nonfinite policy output')
                    pipeline[day+c['lead_time']]=pipeline.get(day+c['lead_time'],0.)+order
                stock,h,b=step(stock,0.,series[t],c['holding_cost'],c['stockout_cost']);hold+=h;short+=b
                traces.append(dict(policy=name,sku_index=sku,day=day,demand=series[t],arrival=arrivals,
                                   order=order,inventory=stock,holding_cost=h,stockout_cost=b))
            rows.append(dict(policy=name,sku_index=sku,holding_cost=hold,stockout_cost=short,total_cost=hold+short))
    return rows,traces


def base_policies(config):
    c=config
    return [('OPT',None),('Normal',lambda history,sku,stock,remaining:normal_order(
                history.mean(),history.std(),c['review_period'],c['lead_time'],0.,stock,c['holding_cost'],c['stockout_cost'])),
            ('Gamma',lambda history,sku,stock,remaining:gamma_order(
                history.mean(),history.std(),c['review_period'],c['lead_time'],stock,c['holding_cost'],c['stockout_cost']))]


def neural_policy(model,config,sku_count,mode,channel=0):
    def policy(history,sku,stock,remaining):
        order,daily,lead,total=predict(model,history,sku,sku_count,stock,config)
        if mode=='e2e':return order
        if mode=='bm1':return bm1_order(total[channel],stock)
        return bm2_order(daily[:,channel],config['lead_time'],config['review_period'],stock,
                         config['holding_cost'],config['stockout_cost'])
    return policy


def gbm_policy(model,config,sku_count):
    def policy(history,sku,stock,remaining):
        inputs,scale=features(history,sku,sku_count,stock,config)
        x=gbm_features(dict(inputs=inputs))
        # Booster.predict avoids sklearn feature-name warnings for array input.
        return max(float(model.booster_.predict(x[None],num_threads=2)[0])*scale,0.)
    return policy


def select_channel(model,sales,config,mode):
    """Choose quantile on VALIDATION trajectories, never the test outcomes."""
    end=int(sales.shape[1]*config['validation_end_fraction'])
    start=end-config['evaluation_days'];values=[]
    # Mean and the paper's six quantiles; median is for Figure 2(a) only.
    candidates=[0]+[i+1 for i,q in enumerate(config['quantiles']) if q!=.5]
    for channel in candidates:
        rows,_=simulate(sales,config,[(mode,neural_policy(model,config,len(sales),mode,channel))],start=start)
        values.append((np.mean([r['total_cost'] for r in rows]),channel))
    return min(values)[1],values
