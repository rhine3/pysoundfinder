# Very basic tests

import pytest as pyt
import simulate as simu
import numpy as np

def test_calc_sos_humidity_not_implemented():
    with pyt.raises(NotImplementedError):
        simu.calc_sos(temp_celsius=1, humidity=5)
        
def test_calc_sos_zero_celsius():
    assert(simu.calc_sos(temp_celsius = 0) == 331.3)
    

def test_compute_toa_standard():
    assert(
        simu.compute_toa(
            np.array((0,0)), 
            np.array((3,4)), 
            simu.calc_sos()
        ) == 0.01456814386554968)
    
    
def test_simulate_distance():
    assert(
        simu.simulate_dist(
            coords_list = [(0, 0), (0, 30), (30, 0), (30, 30)],
            temp_c = 20.0,
            desired_spot = (4, 16)
        ) == [
            0.04805279674148424,
            0.04242307528961981,
            0.08894922601588229,
            0.08603842659634627]
    )

