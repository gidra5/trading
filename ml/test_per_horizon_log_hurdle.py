"""Independent selection and full-copy transfer checks for the corrected run."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from train_per_horizon_log_hurdle import (HORIZONS,horizon_improvements,make_triplet,
    load_combined,payload,sha,verified)


class PerHorizonLogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_each_horizon_can_improve_while_average_worsens(self):
        best=[{'score':1.},{'score':2.},{'score':3.},{'score':4.}]
        np.testing.assert_array_equal(horizon_improvements([.9,2.,10.,3.9],best),[True,False,False,True])
        with self.assertRaises(ValueError): horizon_improvements([np.nan]*4,best)

    def test_actual_retained_sign_zero_transfer_and_independent_copies(self):
        import json
        from neural_sign_size_forecast import StructuredSpecialist
        from train_ternary_sign_size_forecast import singleton_state
        source=json.loads(Path('data/training/runs/stochastic-crra-hurdle-nll-v1/selection.json').read_text())
        reference=verified(source['horizons'][0]['pretrained']['sign'])
        mean,std=reference['inputMean'].numpy(),reference['inputStd'].numpy()
        scales=np.array([[.0001,.0002],[.002,.003],[.004,.005],[.01,.02]])
        size=StructuredSpecialist('size',torch.from_numpy(mean),torch.from_numpy(std),HORIZONS)
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'size.pt'
            torch.save(payload(size,'size',1,mean,std,scales,'test',{}),file)
            size_record={'checkpoint':str(file),'sha256':sha(file)}
            copies=[]
            for j in (0,3):
                model,_=make_triplet(source['horizons'][j],size_record,mean,std,scales,j)
                copies.append(model)
                self.assertEqual(model.horizons,(HORIZONS[j],))
                for name in ('zero','sign'):
                    original=verified(source['horizons'][j]['pretrained'][name])
                    expected=singleton_state(original['modelState'],j,1)
                    for k,v in getattr(model,name).state_dict().items():
                        torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
                expected=singleton_state(size.state_dict(),j,2)
                for k,v in model.size.state_dict().items(): torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            self.assertFalse({p.data_ptr() for p in copies[0].parameters()} & {p.data_ptr() for p in copies[1].parameters()})
            combined=Path(directory)/'combined.pt'
            torch.save(payload(copies[0],'combined',2,mean,std,scales[:1],'test',{}),combined)
            loaded=load_combined({'checkpoint':str(combined),'sha256':sha(combined)},torch.device('cpu'))
            for k,v in loaded.state_dict().items(): torch.testing.assert_close(v,copies[0].state_dict()[k],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
