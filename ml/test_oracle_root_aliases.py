import unittest
import numpy as np
from oracle_root_aliases import exact_root_aliases


class RootAliasTests(unittest.TestCase):
    def test_exact_aliases_preserve_every_state_and_first_descriptor(self):
        states=np.array([[100.,0.,10.,9.],[50.,5.,10.,9.],[100.,0.,10.,9.],
            [50.,5.,10.,9.],[100.,0.,10.,8.]])
        before=states.copy(); selected,inverse=exact_root_aliases(states)
        np.testing.assert_array_equal(selected,[0,1,4])
        np.testing.assert_array_equal(inverse,[0,1,0,1,2])
        np.testing.assert_array_equal(states[selected][inverse],states)
        # Scattering preserves the original argmax tie break, even if the
        # unique values were evaluated in a different batch.
        unique_values=np.array([2.,3.,1.]); scores=unique_values[inverse]
        self.assertEqual(int(scores.argmax()),1)
        np.testing.assert_array_equal(states,before)

    def test_other_wealth_orders_and_float_bits_are_not_merged(self):
        state=np.array([100.,0.,10.,9.,1.,8.,5.])
        states=np.tile(state,(5,1))
        states[1,0]=101.; states[2,5]=8.01; states[3,6]=6.; states[4,1]=-0.
        selected,inverse=exact_root_aliases(states)
        np.testing.assert_array_equal(selected,np.arange(5))
        np.testing.assert_array_equal(inverse,np.arange(5))

    def test_invalid_inputs(self):
        for states in (np.empty((0,4)),np.ones((1,4),dtype=np.float32),np.array([[np.nan]])):
            with self.assertRaises(ValueError): exact_root_aliases(states)


if __name__=='__main__': unittest.main()
