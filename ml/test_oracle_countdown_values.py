import unittest
from types import SimpleNamespace
import numpy as np
from audit_oracle_countdown_values import unchanged_tail


class CountdownTailTests(unittest.TestCase):
    def test_fresh_root_does_not_change_retained_backup_inputs(self):
        anchor=SimpleNamespace(origin_seconds=100.)
        for name in ('prices','counts','features','price_pairs'):
            setattr(anchor,name,np.arange(24).reshape(6,4))
        for name in ('destinations','sampled_prices'):
            setattr(anchor,name,np.arange(20).reshape(5,4))
        for offset in range(5):
            cloud=SimpleNamespace(origin_seconds=100.+offset)
            for name in ('prices','counts','features','price_pairs','destinations','sampled_prices'):
                value=getattr(anchor,name)[offset:].copy();value[0]=-7
                setattr(cloud,name,value)
            unchanged_tail(anchor,cloud,offset)
            cloud.prices[-1,0]+=1
            with self.assertRaisesRegex(ValueError,'Retained tail changed: prices'):
                unchanged_tail(anchor,cloud,offset)
        cloud.origin_seconds+=1
        with self.assertRaisesRegex(ValueError,'timestamp'):unchanged_tail(anchor,cloud,4)


if __name__=='__main__':unittest.main()
