import pytest
import pandas as pd
import pandas.util.testing as pdt
import numpy as np
import numpy.testing as npt
import os
import csv
import pysoundfinder as pysf
import simulate as simu


############################################################
###### Tests of actual localization abilities of pysf ######
############################################################


def test_localize_on_top_of_recorder(simulated_times_df):
    '''
    Test pysf.localize_sound() for sound at (0, 0).
    Asserts accuracy to 6 decimals in each coordinate.
    '''
    
    recorder_list = [(0, 0), (0, 30), (30, 0), (30, 30)]
    true_position = recorder_list[0]
    temp_c = 20.0
    
    positions, times, temps = simulated_times_df(
        sound = true_position,
        recorders = recorder_list,
        temp = temp_c)
    
    [x, y, s] = pysf.localize_sound(
        positions,
        times,
        temps.temp)

    est_position = [x[0], y[0]]
    
    npt.assert_almost_equal(est_position, true_position, decimal=6)
    
    
def test_localize_equal_distances(simulated_times_df):
    '''
    Test pysf.localize_sound() for sound at (15, 15).
    Asserts accuracy to 6 decimals in each coordinate.
    '''
    
    recorder_list = [(0, 0), (0, 30), (30, 0)]
    true_position = (15, 15)
    temp_c = 20.0
    
    positions, times, temps = simulated_times_df(
        sound = true_position,
        recorders = recorder_list,
        temp = temp_c)
    
    [x, y, s] = pysf.localize_sound(
        positions,
        times,
        temps.temp)

    est_position = [x[0], y[0]]
    
    npt.assert_almost_equal(est_position, true_position, decimal=6)
    
    
def test_localize_3d(simulated_times_df):
    '''
    Test pysf.localize_sound() for sound at (4, 2, 20).
    Asserts accuracy to 6 decimals in each coordinate.
    '''
    
    recorder_list = [(0, 0, 2), (0, 30, 3), (30, 0, 3), (30, 30, 3.0)]
    true_position = (4, 2, 20)
    temp_c = 20.0
    
    positions, times, temps = simulated_times_df(
        sound = true_position,
        recorders = recorder_list,
        temp = temp_c)

    
    [x, y, z, s] = pysf.localize_sound(
        positions,
        times,
        temps.temp)

    est_position = [x[0], y[0], z[0]]
    
    # TODO: fix this
    npt.assert_almost_equal(est_position, true_position, decimal=6)
    

    
    

def test_localize_2d_grid(simulated_times_df):
    '''
    Test pysf.localize_sound for a 2d grid of points
    on and around a 30m^2 plot. Asserts accuracy to 
    6 decimals in each coordinate.
    '''
    
    
    recorder_list = [(0, 0), (0, 30), (30, 0), (30, 30)]
    temp_c = 20.0
    grid_points = [(x, y) for x in range(-10, 40, 2)
                          for y in range(-10, 40, 2)]
    
    failed_list = []
    
    for point in grid_points:
        true_position = point

        positions, times, temps = simulated_times_df(
            sound = true_position,
            recorders = recorder_list,
            temp = temp_c)

        [x, y, s] = pysf.localize_sound(
            positions,
            times,
            temps.temp)

        est_position = [x[0], y[0]]
        
        try:
            npt.assert_almost_equal(est_position, true_position, decimal=6)
        except AssertionError:
            failed_list.append((true_position, est_position))
    
    # Print information about the failures if the failed_list contains anything
    for i, failure in enumerate(failed_list):
        print("Failure {}:".format(i+1))
        print("  True position: {} ".format(failure[0]))
        print("  Estimated position: {} ".format(failure[1]))
        
    assert(len(failed_list) == 0)


        

def test_localize_3d_grid(simulated_times_df):
    '''
    Test pysf.localize_sound for a 3d grid of points
    on a 30m^2 x 16m high plot. Asserts accuracy to 
    4 decimals in each coordinate.
    
    Note: sounds with one component exactly at 0 
    aren't localized well for 4 recorders perfectly 
    at the corners of a 30x30 plot. This is a highly 
    unlikely situation, though; even a 0.01m
    offset is enough for near perfect localization.
    
    Also note that if all of the recorders are on 
    the same plane, 3D localization fails frequently.
    Even a +- 0.2m difference in recorder heights is
    enough for highly accurate localization.
    '''
    
    
    recorder_list = [(0, 0, 3), (0, 30, 3.2), (30, 0, 3.3), (30, 30, 3.4)]
    temp_c = 20.0
    grid_points = [(x, y, z) for x in np.arange(0.01, 30, 2)
                             for y in np.arange(0.01, 30, 2)
                             for z in np.arange(0.01, 15, 2)]
    
    failed_list = []
    
    for point in grid_points:
        true_position = point

        positions, times, temps = simulated_times_df(
            sound = true_position,
            recorders = recorder_list,
            temp = temp_c)

        [x, y, z, s] = pysf.localize_sound(
            positions,
            times,
            temps.temp)
    
        est_position = (x[0], y[0], z[0])
        
        
        try:
            npt.assert_almost_equal(est_position, true_position, decimal=4)
        except AssertionError:
            failed_list.append((true_position, est_position))
        
        
    # Print information about the failures if the failed_list contains anything
    for i, failure in enumerate(failed_list):
        pass
        print("Failure {}:".format(i+1))
        print("  True position: {} ".format(failure[0]))
        print("  Estimated position: {} ".format(failure[1]))
    print("Number failed: {}".format(len(failed_list)))
    assert(len(failed_list) == 0)

        
#####################################################
###### Test input/output functionality of pysf ######
#####################################################



def test_df_creation_from_files(positions_df, times_df, temps_df, tmpdir):
    '''
    Compare the correct DF formats (as created in fixtures)
    with the DF formats generated by PySF
    '''
    
    posi_csv_path = tmpdir.join('positions.csv')
    time_csv_path = tmpdir.join('times.csv')
    
    positions_txt = [
        ['recorder','x','y'],
        ['r1','0','0'],
        ['r2','0','30'],
        ['r3','30','0'],
        ['r4','30','30']]
    
    times_txt = [
        ['idx','r1','r2','r3','r4','temp'],
        ['0','0.0','0.08740886319329808','0.08740886319329808','0.12361479979957658','20.0']]
    
    with open(posi_csv_path, 'w+') as csvfile:
        writer = csv.writer(csvfile)
        for row in positions_txt:
            writer.writerow(row)
            
    with open(time_csv_path, 'w+') as csvfile:
        writer = csv.writer(csvfile)
        for row in times_txt:
            writer.writerow(row)

    pysf_positions, pysf_times, pysf_temps = pysf.dfs_from_files(posi_csv_path, time_csv_path)
    
    test_positions = positions_df(
        [(0, 0),
        (0, 30),
        (30, 0),
        (30, 30)])
    
    
    test_times = times_df([0.0,0.08740886319329808,0.08740886319329808,0.12361479979957658])
    
    test_temps = temps_df(20)
 
    print(pysf_positions, test_positions)
    
    pdt.assert_frame_equal(pysf_temps, test_temps)
    pdt.assert_frame_equal(pysf_positions, test_positions)#, check_names=False)
    pdt.assert_frame_equal(pysf_times, test_times)



######################################################
###### Fixtures for creating dataframes for     ######
###### recorder positions, time delay, and temp ######
######################################################


@pytest.fixture
def simulated_times_df(positions_df, times_df, temps_df):
    '''
    Simulate one TDOA set given recorder & sound positions
    
    Given the position of a sound source, the
    positions of recorders in an array, and
    the temperature, computes time of arrivals
    at the recorders.
    
    Example usage:
        simulated_times_df(
            sound = [3, 4],
            recorders = [
                [0, 0],
                [0, 30],
                [30, 0]
            ],
            temp = 20.0
        )
        
        
    '''
    
    # TODO: convert simu.simulate_dist, positions_df, etc. to lists of coords
    
    def _get_args(sound, recorders, temp):
        
        positions = positions_df(recorders)
        
        delays = simu.simulate_dist(
            coords_list = recorders,
            desired_spot = sound,
            temp_c = temp,
            print_results = False)
        
        times = times_df(delays)
        
        temp = temps_df(temp)

        return positions, times, temp

    return _get_args
        


@pytest.fixture
def positions_df():
    '''
    Return a dataframe of positions
    
    Return a dataframe of positions in the correct
    format. Input should be a list of either 
    3 or 4 sets of 2D or 3D coordinates.
    
    Example usage:
        positions_df(
            [(0,0),
            (0,30),
            (30, 0), 
            (30, 30)])
    '''
    
    #TODO: ADD SUPPORT FOR > 4 RECORDERS
    
    def _get_recorders(recorder_list):
        
        # Assert we were given 3 or 4 tuples of coordinates
        assert(len(recorder_list) in [3, 4]) 

        # Create a dictionary associating recorder names
        # with coordinates given from arguments
        r1 = recorder_list[0]
        r2 = recorder_list[1]
        r3 = recorder_list[2]
        try:
            r4 = recorder_list[3]
            pos_dict = {'r1':r1, 'r2':r2, 'r3':r3, 'r4':r4}
        except:
            pos_dict = {'r1':r1, 'r2':r2, 'r3':r3}


        # Assert we were given 2D or 3D recorder coordinates;
        # all coordinate tuples must be same length
        dimensions = len(r1)
        assert(dimensions in [2, 3])
        for key in pos_dict:
            assert len(pos_dict[key]) == dimensions

        # Get correct dimensions for index
        if dimensions == 2: idx = ['x', 'y']
        else: idx = ['x', 'y', 'z']

        positions_df = pd.DataFrame(
            pos_dict, 
            index = idx,
            dtype='float64').T
        positions_df.index.names = ['recorder']

        return positions_df
    
    return _get_recorders


@pytest.fixture
def times_df():
    
    '''
    Arguments should be a list of the columns of the pandas df.

    Example usage for 4 recorders, 1 sound:

        times_df([0.01, 0, 0.001])


    Or for 4 recorders, 3 sounds, note that
    the input must be a list of lists:

    times_df(
        [[0, 0.01, 0],
        [0.001, 0.02, 0.03],
        [0.02, 0, 0.01],
        [0, 0.002, 0.1]]
    )

    or 
    )
    '''
    
    def _get_times(*args):


        # Assert our input looks good
        assert len(args) == 1, 'Must pass only one argument, a list of delays'
        num_recorders = len(args[0])
        assert (num_recorders in [3, 4]), 'Must pass a list of delays for 3 or 4 recorders'

        # See if we were given a single set of delays, or multiple sets for multiple sounds
        if type(args[0][0]) == list: num_sounds = len(args[0][0])
        else: num_sounds = 1

        # Make the initial dict of at least 3 delays
        delay_dict = {
            'r1':args[0][0],
            'r2':args[0][1],
            'r3':args[0][2]}

        # Try to add args[0][3] to the dict,though it may not exist
        try: delay_dict['r4'] = args[0][3]
        except: pass

        times_df = pd.DataFrame(
            delay_dict, 
            index = pd.RangeIndex(start=0, stop=num_sounds, step=1),
            dtype='float64')
        
        # 
        times_df.index = times_df.index.astype(str)


        return times_df
    
    return _get_times


@pytest.fixture
def temps_df():
    
    def _get_temp(temp_c):

        if type(temp_c) == list: num_temps = len(temp_c)
        else: num_temps = 1

        temp_df = pd.DataFrame(
            {'temp':temp_c}, 
            index = pd.RangeIndex(start=0, stop=num_temps, step=1),
            dtype='float64') 
        temp_df.index = temp_df.index.astype(str)

        return temp_df
    
    return _get_temp



######################################################################
###### Tests of fixtures to ensure they produce the desired dfs ######
######################################################################

def test_simulated_times_df(simulated_times_df, positions_df, times_df, temps_df):
    fixt_posi, fixt_time, fixt_temp = simulated_times_df(
            sound = [3, 4],
            recorders = [
                [0, 0],
                [0, 30],
                [30, 0]
            ],
            temp = 20.0
        )
    
    desired_posi = positions_df(
        [[0, 0], [0, 30], [30, 0]]
    )
    
    desired_time = times_df(
        [0.01456814386554968, 0.07625696263183752, 0.07952658868254762])
    
    desired_temp = temps_df(20.0)

    pdt.assert_frame_equal(fixt_posi, desired_posi)

    pdt.assert_frame_equal(fixt_time, desired_time)

    pdt.assert_frame_equal(fixt_temp, desired_temp)


    
def test_fixture_positions_df_3d_recorders(positions_df):
 
    fixture_3d_positions = positions_df(
        [(0, 0, 0),
        (0, 30, 1),
        (30, 0, 2),
        (30, 30, 3)])
    
    desired_3d_positions = pd.DataFrame(
        {'r1':(0, 0, 0),
         'r2':[0, 30, 1],
         'r3':[30, 0, 2],
         'r4':[30, 30, 3]},
        index = ['x','y', 'z'],
        dtype='float64').T
    desired_3d_positions.index.names = ['recorder']

    pdt.assert_frame_equal(fixture_3d_positions, desired_3d_positions)
    


def test_fixture_positions_df_2d_recorders(positions_df):
 
    fixture_2d_positions = positions_df(
        [[0, 0],
        [0, 30],
        (30, 0),
        (30, 30)])
    
    desired_2d_positions = pd.DataFrame(
        {'r1':(0, 0),
         'r2':[0, 30],
         'r3':[30, 0],
         'r4':[30, 30]},
        index = ['x','y'],
        dtype='float64').T
    desired_2d_positions.index.names = ['recorder']

    pdt.assert_frame_equal(fixture_2d_positions, desired_2d_positions)
    
    
def test_fixture_times_df_one_sound(times_df):
 
    fixture_times = times_df([0.01, 0, 0.003])
    
    desired_times = pd.DataFrame(
        {'r1':0.01,
         'r2':0,
         'r3':0.003},
        index = ['0'],
        dtype='float64')
    
    print(fixture_times, desired_times)

    pdt.assert_frame_equal(fixture_times, desired_times)
    

def test_fixture_times_df_two_sounds(times_df):
    
    fixture_times = times_df([[0, 0], [0.001, 0.002], [0.10, 0.09]])
    
    desired_times = pd.DataFrame(
        {'r1':[0, 0],
         'r2':[0.001, 0.002],
         'r3':[0.10, 0.09]},
        index = ['0', '1'],
        dtype='float64')
    
    print(fixture_times, desired_times)

    pdt.assert_frame_equal(fixture_times, desired_times)
    


def test_fixture_temps_df(temps_df):
    
    fixture_temp = temps_df(20)
    
    desired_temp = pd.DataFrame(
        {'temp':20.0},
        index = ['0'],
        dtype='float64')
    desired_temp.index = desired_temp.index.astype(str)
    
    print(fixture_temp, desired_temp)

    pdt.assert_frame_equal(fixture_temp, desired_temp)
    
    

