"""Figure 1 topology with an explicit multi-horizon quantile RNN decoder.

The paper does not specify decoder internals, auxiliary weights or loss norms.
Those implementation choices are documented instead of claiming author code.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MQRNN(nn.Module):
    """LSTM encoder + global context decoder + shared horizon-local decoder."""
    def __init__(self, hidden, basic_size, horizon, channels, dropout):
        super().__init__()
        self.horizon=horizon
        self.encoder=nn.LSTM(1+basic_size,hidden,batch_first=True)
        self.global_decoder=nn.Sequential(nn.Linear(hidden+basic_size,hidden*(horizon+1)),nn.ReLU(),nn.Dropout(dropout))
        self.local_decoder=nn.Sequential(nn.Linear(2*hidden+1,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,channels))
        self.total_head=nn.Sequential(nn.Linear(hidden+basic_size,hidden),nn.ReLU(),nn.Linear(hidden,channels))

    def forward(self, history, basic):
        inputs=torch.cat([history,basic[:,None,:].expand(-1,history.shape[1],-1)],-1)
        _,(states,_)=self.encoder(inputs)
        encoding=states[-1]
        context=self.global_decoder(torch.cat([encoding,basic],-1)).reshape(len(history),self.horizon+1,-1)
        shared=context[:,:1].expand(-1,self.horizon,-1)
        time=torch.linspace(0,1,self.horizon,device=history.device)[None,:,None].expand(len(history),-1,-1)
        forecast=F.softplus(self.local_decoder(torch.cat([context[:,1:],shared,time],-1)))
        total=F.softplus(self.total_head(torch.cat([encoding,basic],-1)))
        return encoding,forecast,total


class InventoryNetwork(nn.Module):
    def __init__(self, config, sku_count):
        super().__init__()
        c=config; self.config=c
        self.df=MQRNN(c['hidden'],sku_count,c['lead_time']+c['review_period'],len(c['quantiles'])+1,c['dropout'])
        vlt=[]; width=1+sku_count
        for next_width in c['vlt_layers']:
            vlt.extend([nn.Linear(width,next_width),nn.ReLU(),nn.Dropout(c['dropout'])]);width=next_width
        self.vlt=nn.Sequential(*vlt);self.vlt_output=nn.Linear(width,1)
        self.fusion=nn.ModuleList()
        width=c['hidden']+width+1 # review period injected at first fusion layer
        for index,next_width in enumerate(c['fusion_layers']):
            self.fusion.append(nn.Linear(width+(1 if index==len(c['fusion_layers'])-1 else 0),next_width))
            width=next_width
        self.output=nn.Linear(width,1)
        self.dropout=nn.Dropout(c['dropout'])
        # Table B.1: all trainable weights N(0,0.01), zero biases.
        for name,param in self.named_parameters():
            nn.init.zeros_(param) if 'bias' in name else nn.init.normal_(param,0,c['init_std'])

    def forward(self, history, basic, vlt_features, review, inventory):
        encoded,forecast,total=self.df(history,basic)
        vlt_state=self.vlt(torch.cat([vlt_features,basic],-1))
        lead=F.softplus(self.vlt_output(vlt_state)).squeeze(-1)
        state=torch.cat([encoded,vlt_state,review],-1)
        for index,layer in enumerate(self.fusion):
            if index==len(self.fusion)-1:
                state=torch.cat([state,inventory],-1)
            state=self.dropout(F.relu(layer(state)))
        order=F.softplus(self.output(state)).squeeze(-1)
        return order,forecast,lead,total


def forecast_loss(prediction,target,quantiles):
    qs=torch.tensor(quantiles,device=target.device)
    error=target[...,None]-prediction[...,1:]
    return F.mse_loss(prediction[...,0],target)+torch.maximum(qs*error,(qs-1)*error).mean()


def loss(outputs,order_target,future,lead_target,config,mode='e2e'):
    order,daily,lead,total=outputs
    # BM1 has a total-demand head, as specified in the paper's endnote.
    demand_loss=forecast_loss(daily,future,config['quantiles']) if mode!='bm1' else forecast_loss(total,future.sum(-1),config['quantiles'])
    vlt_loss=F.mse_loss(lead,lead_target)
    if mode=='e2e':
        return F.mse_loss(order,order_target)+config['auxiliary_weights'][0]*demand_loss+config['auxiliary_weights'][1]*vlt_loss
    return demand_loss+vlt_loss
