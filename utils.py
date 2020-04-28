'''
Utilities for testing localization

For example usage, run utils.example().
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate_dist
import pysoundfinder as pysf



################################################################################
########## UTILITIES FOR SETTING UP A GRID OF SOUNDS
################################################################################


def find_min_max(df):
    x_min = min(df['x'])
    x_max = max(df['x'])
    y_min = min(df['y'])
    y_max = max(df['y'])
    mini = min(x_min, y_min)
    maxi = max(x_max, y_max)
    
    return mini, maxi


def make_delay_grid(
        r_locs,
        heights = [0],
        drop_locs = False,
        spacing = 1,
        temperature = 20.0
):
    '''
    Make grid of sound delays
    
    Inputs:
        r_locs: 
            Pandas DF of recorder locations. 
            Columns (r1, r2, r3, r4); rows (x, y, z)
        heights:
            List of heights from which sound source should emanate above 
            recorder plane. Set to [0] for a sound on the recorder plane.
        drop_locs:
            Whether or not to drop sound sources at recorder locs
    
    Returns:
        a list of lists.
        Inner lists: 
    '''
    
    ### Specify sound source locations from recorder locations ###
    mini, maxi = find_min_max(r_locs)
    
    coords = np.arange(start=np.ceil(mini)+1, stop=np.ceil(maxi), step = spacing)

    xx, yy, zz = np.meshgrid(
        coords,
        coords,
        heights
    )

    # Create a list of points
    points_list = list(zip(xx.flatten(), yy.flatten(), zz.flatten()))

    # Convert it to a dataframe of sound locations
    df = pd.DataFrame(points_list).T
    df.index = ['x', 'y', 'z']
    s_locs = df.T
    
    
    ### Drop sounds from recorder locations ###
    if drop_locs:
        indices = []
        
        # Get coordinates of recorders
        for r_name in r_locs.index:
            recorder = r_locs.loc[r_name]
            my_child = s_locs.index[(s_locs['x']==recorder.loc['x']) & (s_locs['y']==recorder.loc['y'])].tolist()
            indices.append(my_child)
        to_drop = list(np.array(indices).flatten())
        
        # Drop
        s_locs = s_locs.drop(index=to_drop)
        print("Sounds dropped:", to_drop)
        
    ### Make delays ###
    r_tuples = [tuple(df[1]) for df in r_locs.iterrows()]
    delay_list = []
    for sound in s_locs.iterrows():
        sound_tuple = tuple(sound[1])
        delays = simulate_dist(
            recorder_coords = r_tuples, 
            source_coords = sound_tuple,
            temp_c = temperature, 
            print_results = False
        )
        delay_list.append(delays)
    
    #return delay_list
    ### Format delays nicely ###
    # This DF also needs to be reindexed
    all_sounds = pd.DataFrame(delay_list).T.set_index(r_locs.index).T
    all_sounds[0:10]
        
    return (s_locs, all_sounds)



################################################################################
########## UTILITIES FOR CREATING ERRORS
################################################################################

def jitter_delays(
    all_sounds,
    second_error,
    uniform_errs = True,
    seed = 43110
):
    
    '''
    Jitter delays and return relative delay df
    
    Jitter delays such that each recorder has a different error in 
    time of arrival. (Does NOT draw a separate error for each delay,
    only for each recorder. The same error is used for all delays 
    arriving at a specified recorder)
    
    Inputs:
        all_sounds: df of delays:
            columns: ['r1', 'r2', ...]
            rows: [0, 1, 2, ...]
        second_error: desired +- error in seconds
        uniform_errs: whether to draw errors from uniform distribution
            if True: errors drawn from Uniform(-seconds_error, seconds_error)
            if False: errors drawn from Normal(mean=0, sd=seconds_error)
        seed: random seed if desired
    
    Returns: 
        rel_sounds: df of delays given relative errors.
    '''
    
    #np.random.seed(seed)
    
    # Instead of adding a different error for each sound, add a single error for each recorder.
    err_sounds = all_sounds.copy()
    num_recs = err_sounds.shape[1]

    for recorder in err_sounds.columns:
        if uniform_errs:
            unif_err = np.random.uniform(low=-second_error, high=second_error)
            err_sounds[recorder] += unif_err
        else:
            norm_err = np.random.normal(loc=0.0, scale=second_error, size=None)
            err_sounds[recorder] += norm_err

    # Create relative delays df
    rel_sounds = err_sounds.copy()
    for i in range(rel_sounds.shape[0]):
        rel_sounds.iloc[i] = rel_sounds.iloc[i] - min(rel_sounds.iloc[i])
    
    return rel_sounds


def make_jitter_trials(
    df,
    amt,
    trials = 5,
):
    '''
    Return a list of jitter dataframes
    
    Inputs:
        df: a delay dataframe
        amt: amount to jitter
            error will be drawn from Uniform(-amt, amt)
        trials: number of jitter trials
        
    Returns:
        jf: a list of dataframes
            len(jf) = trials
        
    '''
    
    jf = [None] * trials
    
    for i in range(trials):
        jf[i] = jitter_delays(
            df,
            second_error = amt,
            uniform_errs=True,
            seed=None)

    return jf



################################################################################
########## UTILITIES FOR LOCALIZATION
################################################################################


def localize_pysf(
    rel_sounds,
    r_locs,
    invert_alg = 'gps',
    center_array = True,
    use_sos_selection = False,
    temperature = 20.0):
    '''
    Use PySoundFinder to localize many sounds
    
    Inputs:
        rel_sounds (pandas DF): relative TOAs of sounds
            columns are recorder names
            rows are sound indices
        r_locs (pandas DF): recorder locations
            columns are x, y, and optionally, z
            rows are recorder names as in rel_sounds
        invert_alg: algorithm to invert matrix (see pysoundfinder)
        center_array (bool): whether to center array coordinates
            for computational stability prior to localizing sounds
        use_sos_selection (bool): whether to select solution
            based on sum of squares (original Sound Finder behavior)
        temperature (float): temperature in Celsius
    '''
    
    # Localize sounds in rel_sounds
    localized = []
    for sound_num in rel_sounds.index:
        delays = rel_sounds.T[sound_num]

        localized.append(pysf.localize_sound(
            positions = r_locs,
            times = delays,
            temp = temperature,
            invert_alg = 'gps',
            center = center_array,
            pseudo = not use_sos_selection
        ))
        
    # Return dataframe in desired format
    # TODO: Currently, reshape is necessary in 3D case to bring results 
    # into an (A, B) shaped matrix instead of an (A, B, 1) shaped matrix.
    # This doesn't happen in 2D case. Worth inspecting. This try/except is a quick patch.
    try:
        df = pd.DataFrame(np.array(localized).reshape(rel_sounds.shape), index=rel_sounds.index)
    except ValueError:
        return localized
    if df.shape[1] == 4:
        df.columns = ['x', 'y', 'z', 'r']
    elif df.shape[1] == 3:
        df.columns = ['x', 'y', 'r']
    else:
        print('Warning: results are not expected dimension (3 or 4 columns)')
    return df


################################################################################
########## UTILITIES FOR FINDING ERROR
################################################################################

def find_error(a, b):
    return np.linalg.norm(a - b)


def calc_errors_df(original_locs, localized_df):
    '''
    Calculate 3d errors where estimated locs are a df
    
    Inputs:
        original_locs: pandas df
            columns: ['x', 'y', 'z'] - location of delay
            rows: index of delay
        localized_df: pandas df
            columns: ['x', 'y'] - estimated location
    '''
    
    errors = original_locs.copy(deep=True)
    errors['estimate-x'] = np.nan
    errors['estimate-y'] = np.nan
    errors['error'] = np.nan

    for i in [int(idx) for idx in localized_df.index]:
        # Get np.arrays of the true and estimate point
        estimate = np.array([localized_df.iloc[i]['x'], localized_df.iloc[i]['y']])
        true = np.array(original_locs.iloc[i])[0:2]

        # Add the error in the row of the DF
        error = find_error(estimate, true)
        errors.loc[i, 'error'] = error
        errors.loc[i, 'estimate-x'] = estimate[0]
        errors.loc[i, 'estimate-y'] = estimate[1]
    
    return errors


def calc_all_errors(original_locs, dfs):
    '''
    Calculate localization errors for many dataframes
    
    Example usage: 
        errors = calc_all_errors(sound_locs, [sos_df, pseudorange_df])
    
    Inputs:
        original_locs: pandas DF of original locations
            columns: ['x', 'y', 'z']
            rows: true location for each sound source
            
        dfs: either a dictionary of lists or a list of lists
    
            - a dictionary associating names of types of experiments with 
          lists of results of estimates for trials of experiments   
            - a list of pandas DFs of location estimate trials:
            
            In either case, inner lists should each contain the same number of DFs:
                [[py_trial_one_df, py_trial_two_df],
                 [r_trial_one_df, r_trial_two_df]]
                 
                {'py_experiments': [py_trial_one_df, py_trial_two_df],
                 'r_experiments': [r_trial_one_df, r_trial_two_df]}
                 
            Even if there is only one trial, dfs should still be a LOL or dict of lists:
                [[py_one_trial_df], [r_one_trial_df]])
                
                {'py_experiments': [py_trial_one_df],
                 'r_experiments': [r_trial_one_df]}
                 
            Each df should have:
                columns: ['x', 'y']
                rows: estimate for each sound source
            
    Returns:
        A list of lists of errors. Shape corresponds to the shape of the
        input list of lists, `dfs`. For example, given the input:
                dfs = [[py_trial_one_df, py_trial_two_df],
                      [r_trial_one_df, r_trial_two_df]]
            The return will be:
                [[py_trial_one_error_df, py_trial_two_error_df],
                 [r_trial_one_error_df, r_trial_two_error_df]]
                 
            Given the input:
                {'py_experiments': [py_trial_one_df, py_trial_two_df],
                 'r_experiments': [r_trial_one_df, r_trial_two_df]}
            The return will be:
                {'py_experiments': [py_trial_one_error_df, py_trial_two_error_df],
                 'r_experiments': [r_trial_one_error_df, r_trial_two_error_df]}
    
    '''
    
    if type(dfs) == list:
        
        trials = len(dfs[0])
        for df in dfs:
            assert(type(df) == list)
            assert(len(df) == trials)
            
        length = len(dfs)
        error_trials = [[None] * trials] * length

        for df_idx, df in enumerate(dfs):
            for trial_idx in range(trials):
                error_trials[df_idx][trial_idx] = calc_errors_df(original_locs, df[trial_idx])

            
    elif type(dfs) == dict:
        keys = list(dfs.keys())
        trials = len(dfs[keys[0]])
        for key in dfs.keys():
            df = dfs[key]
            assert(type(df) == list)
            assert(len(df) == trials)
            
            
        length = len(dfs)
        error_trials = {key: [None]*trials for key in keys}

        for key in keys:
            df = dfs[key]
            for trial_idx in range(trials):
                error_trials[key][trial_idx] = calc_errors_df(original_locs, df[trial_idx])
        
    else:
        raise ValueError('dfs must be either dict or list')
        


    return error_trials

def avg_error(loc_dfs):
    '''
    Given a list of dataframes containing true and 
    estimated locations, plus the error between the 
    estimates, create a new dataframe showing the average
    error between all dataframes in the list.

    Inputs:
        loc_dfs: list of pandas dfs where each df has the format:
            columns: ['x', 'y', 'z', 'estimate-x', 'estimate-y', 'error']
            rows: one for each (x, y, z) point estimate
    
    Returns: 
    '''
    
    from copy import deepcopy
    
    ref_df = deepcopy(loc_dfs[0])[['x', 'y', 'z']]
    
    for idx, df in enumerate(loc_dfs):
        ref_df[f'err_{idx}'] = df['error']
    
    err_cols = [col for col in ref_df.columns if col.startswith('err_')]

    ref_df['avg_error'] = ref_df[err_cols].mean(axis=1)
    ref_df['std_dev'] = ref_df[err_cols].std(axis=1)
    
    return ref_df[['x', 'y', 'z', 'avg_error', 'std_dev']]



################################################################################
########## UTILITIES FOR PLOTTING
################################################################################

def make_axis(df, ax, color_max, title, recs, std_dev=False):
    ax.set(aspect=1)
    ax.set_title(title)
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    # possibly use imshow here, but be careful; y axis is flipped
    if std_dev:
        im = ax.scatter(df['x'], df['y'], c=df['std_dev'], vmin=0, vmax=color_max)
    else:
        im = ax.scatter(df['x'], df['y'], c=df['avg_error'], vmin=0, vmax=color_max)
    im2 = ax.scatter(recs.T['x'], recs.T['y'], c='red')
    return im 

def make_err_plot(
    err_dfs,
    recs,
    ms_error,
    trials,
    fig_size = (15, 20),
    h_space = 0.5,
    vert_max = None,
    std_dev = True,
    sep_by = [0]
):
    
    '''
    Create a plot of several dfs of errors
    
    Inputs:
        err_dfs: a dictionary of dfs to plot, one per row
            - keys: name to be displayed on plot
            - values: dataframe of errors with columns ['x', 'y', 'z', 'errors'] 
              representing true location (x, y, z) and error (errors)
        recs: df of recorder locations
        ms_error: error for uniform jitter, presumed to be in ms
        sep_by: indexers for err_df['z'], one per column; presumed to be in meters
    
    '''

    # Determine bounds of color bar if not given
    if not vert_max:
        err_max = 1 #top of colorbar at minimum
        # Find highest error
        for df_key in err_dfs.keys():
            df_high = np.max(err_dfs[df_key]['avg_error'])
            if (df_high > err_max):
                err_max = df_high
        print('maximum average error:', err_max)
        #Use whichever is smallest, highest error or 25
        vert_max = min(err_max, 25)
    
    
    rows = len(err_dfs)
    cols = len(sep_by)
    if std_dev == True:
        assert(cols == 1) #only do this for one height
        cols = 2  #one for avg, one for std dev
    
    fig, axes = plt.subplots(
        nrows = rows,
        ncols = cols,
        figsize = fig_size
    )
    
    # For each row/df
    for row, key in enumerate(err_dfs.keys()):
        df = err_dfs[key]
        
        if not std_dev:
            
            # For each column/height (aka sep), make a plot
            for col in range(cols):
                sep = sep_by[col]
                separated = df.loc[df['z'] == sep]

                if rows == 1:
                    
                    if cols == 1:
                        
                        # Can't subset either
                        im = make_axis(separated, axes, vert_max, f'{key}\naverage loc errors ({trials} trials) \nUniform(-{ms_error}ms, {ms_error}ms),\n{sep}m above plane', recs, std_dev=False)
                        
                    else:
                        # Can only subset cols
                        im = make_axis(separated, axes[col], vert_max, f'{key}\naverage loc errors ({trials} trials) \nUniform(-{ms_error}ms, {ms_error}ms),\n{sep}m above plane', recs, std_dev=False)
                    
                else:
                
                    if cols == 1:
                        
                        # Can only subset rows
                        im = make_axis(separated, axes[row], vert_max, f'{key}\naverage loc errors ({trials} trials) \nUniform(-{ms_error}ms, {ms_error}ms),\n{sep}m above plane', recs, std_dev=False)
                    
                    else:
                        # General case--can subset both
                        im = make_axis(separated, axes[row][col], vert_max, f'{key}\naverage loc errors ({trials} trials) \nUniform(-{ms_error}ms, {ms_error}ms),\n{sep}m above plane', recs, std_dev=False)
            
        else:
            sep = sep_by[0]
            separated = df.loc[df['z'] == sep]
            
            # Make average axis
            im = make_axis(separated, axes[row][0], vert_max, f'{key}\naverage loc errors ({trials} trials) \nUniform(-{ms_error}ms, {ms_error}ms),\n{sep}m above plane', recs, std_dev=False)

            # Make std dev axis
            im = make_axis(separated, axes[row][1], vert_max, f'{key}\nstd dev loc errors ({trials} trials) \nUniform(-{ms_error}ms, {ms_error}ms),\n{sep}m above plane', recs, std_dev=True)
    
    #fig.subplots_adjust(hspace = h_space)
    cb = fig.colorbar(im, ax=axes.ravel().tolist())
    cb.ax.set_title('meters error')
    
    return fig 


def example():
    print('''
Example usage:

```
import utils
import pandas as pd
import numpy as np


# Create grid of 3D delays

temp = 20.0

### Dataframe of recorders on plane
r_locs_3d = pd.DataFrame(
    {
        'r1': (0, 0, 0),
        'r2':(0, 25, 0),
        'r3':(25, 0, 0),
        'r4':(25, 25, 0)
    },
    index = ['x', 'y', 'z']).T

### Heights for each set of delays
heights = np.array([0, 10])

### Create grid of delays
(s_locs, true_delays) = utils.make_delay_grid(r_locs_3d, heights, spacing = 2)



# Add error to delays

### Error distributed Uniform(-0.2, 0.2)
ms_error = 0.2

### Number of trials to run
num_trials = 5

### Add the error
ms_error = 0.20
s_error = ms_error/100
jittered = utils.make_jitter_trials(
    df = true_delays,
    amt = s_error,
    trials = num_trials,
)



# Localize the 3D sounds in 2D

### Take recorder locations to 2D
r_locs = r_locs_3d[['x', 'y']]

### Localize using PYSF
py_est_dict = {}
# All combinations
for c in [True, False]:
    for s in [True, False]:
        for a in ['gps', 'lstsq']:
            key = f'ALGO: {a}, CENTER: {c}, SOS: {s}'
            py_est = [None] * num_trials

            # Localize each trial in the jitter array
            for idx, jitter in enumerate(jittered):
                py_trial = utils.localize_py_new(
                    rel_sounds = jittered[idx],
                    r_locs = r_locs,
                    invert_alg = a,
                    center_array = c,
                    use_sos_selection = s,
                    temperature = temp
                )

                py_est[idx] = py_trial

            py_est_dict[key] = py_est


keys = list(py_est_dict.keys())



# Plot error

### Find error
error_types = utils.calc_all_errors(
    original_locs = s_locs,
    dfs = py_est_dict)

### Average error across trials
avg_err_dict = {key: [None] for key in keys}
for key in keys:
    avg_err_dict[key] = utils.avg_error(error_types[key])
    
### Create plot
utils.make_err_plot(
    err_dfs = avg_err_dict,
    recs = r_locs.T,
    ms_error = ms_error,
    trials = num_trials,
#    vert_max = ,
    sep_by = [0, 10],
    std_dev = False,
    fig_size = (10, 30),
    h_space = 1
).show()
```
''')