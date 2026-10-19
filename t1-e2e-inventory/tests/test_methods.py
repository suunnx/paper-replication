import itertools
import json
import unittest
from pathlib import Path
import numpy as np
import torch
from src.inventory import label,normal_order,gamma_order,bm1_order,bm2_order,step
from src.network import InventoryNetwork,loss
from src.data import make_samples,tensor_dataset
from src.experiment import simulate

class MethodsTests(unittest.TestCase):
    def test_theorem_matches_exhaustive_search(self):
        rng=np.random.default_rng(42)
        for _ in range(300):
            d=rng.integers(0,8,size=rng.integers(1,14));stock=int(rng.integers(-10,40))
            h,b=[(1,9),(12,88),(1,1),(9,1)][int(rng.integers(4))]
            def cost(q):
                values=stock+q-np.cumsum(d)
                return np.maximum(values,0).sum()*h+np.maximum(-values,0).sum()*b
            self.assertAlmostEqual(cost(label(d,stock,h,b)),min(cost(q) for q in range(max(0,int(sum(d))-stock)+2)))

    def test_invalid_labels(self):
        for d in [[],[-1],[float('nan')]]:
            with self.assertRaises(ValueError):label(d,0)

    def test_equations_7_to_10(self):
        self.assertEqual(normal_order(2,0,3,2,0,1,1,9),9)
        self.assertEqual(gamma_order(2,0,3,2,1,1,9),9)
        self.assertEqual(bm1_order(10,12),0)
        self.assertEqual(bm2_order([2,2,2,2,2],2,3,1,1,9),9)
        self.assertEqual(step(2,3,7,1,9),(-2,0,18))

    def test_purged_target_splits_and_features(self):
        c=json.loads(Path('configs/paper.json').read_text())
        sales=np.ones((2,457));original=make_samples(sales,c)
        ranges={p:set() for p in original}
        for p,samples in original.items():
            for s in samples:ranges[p].update(range(s['t'],s['end']))
        self.assertFalse(ranges['train'] & ranges['validation'])
        self.assertFalse(ranges['validation'] & ranges['test'])
        self.assertFalse(ranges['train'] & ranges['test'])
        changed=sales.copy();changed[:,c['history']:]=100
        after=make_samples(changed,c)
        for x,y in zip(original['train'][0]['inputs'],after['train'][0]['inputs']):np.testing.assert_array_equal(x,y)
        self.assertNotEqual(original['train'][0]['target'],after['train'][0]['target'])

    def test_network_shapes_and_all_losses_backward(self):
        c=json.loads(Path('configs/paper.json').read_text());c['history']=8;c['hidden']=8
        c['vlt_layers']=[8,4];c['fusion_layers']=[8,8]
        values=make_samples(np.ones((2,100)),c)
        batch=tensor_dataset(values['train'][:3],c).tensors
        for mode in ['e2e','bm1','bm2']:
            model=InventoryNetwork(c,2);outputs=model(*batch[:5])
            self.assertEqual(outputs[1].shape,(3,12,8))
            value=loss(outputs,*batch[5:],c,mode);value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertGreater(model.df.encoder.weight_ih_l0.grad.abs().sum(),0)
        c['fusion_layers']=[8]
        self.assertEqual(InventoryNetwork(c,2)(*batch[:5])[0].shape,(3,))

    def test_continuous_oracle_matches_joint_bruteforce(self):
        c=dict(history=3,lead_time=1,review_period=2,holding_cost=1,stockout_cost=9,evaluation_days=5)
        sales=np.array([[0.,0.,0.,1.,2.,1.,2.,1.]])
        rows,trace=simulate(sales,c,[('OPT',None)])
        best=float('inf')
        for q1,q2 in itertools.product(range(8),repeat=2):
            inv=cost=0.
            for t,d in enumerate(sales[0,3:]):
                inv,h,b=step(inv,q1 if t==1 else q2 if t==3 else 0,d,1,9);cost+=h+b
            best=min(best,cost)
        self.assertEqual(rows[0]['total_cost'],best)
        self.assertEqual(trace[0]['arrival'],0)
        self.assertEqual(trace[1]['arrival'],trace[0]['order'])
        self.assertEqual(trace[-1]['order'],0)
