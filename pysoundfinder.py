import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt

def plot_solution(positions, u):
    # Plot recorders as black circles
    x_coords = positions['x']
    y_coords = positions['y']
    plt.plot(x_coords, y_coords,'ko')

    # Plot solution as a red circle
    plt.plot(u[0], u[1], 'ro')
    
    # Add title showing error
    plt.title('Error: {}'.format(u[-1:][0][0]))

    plt.show()

def lorentz_ip(x1, x2 = 'none', dim=None):
    '''
    Compute Lorentz inner product
    
    Compute Lorentz inner product. For vectors `u` and `v`, the
    Lorentz inner product is defined as
    
        u[0]*v[0] + u[1]*v[1] + u[2]*v[2] - u[3]*v[3]
        
    though in this implementation, u and v can be 4 elements long 
    (for 3d localization) or 3 elements long (for 2d localization).
    x1 and x2 can be np.ndarrays or similar (e.g. pandas series)
    
    Inputs
        x1: vector with shape either (3,) or (4,) 
        x2: vector with same shape as x1
        dim: integer equal to x1.shape[0]-1. Also the number 
            of dimensions in which to localize.
    Returns
        value of Lorentz IP
    '''
    
    '''
    if (type(x1) != np.ndarray) or (type(x2) != np.ndarray):
        print(x1)
        print(type(x1))
        raise ValueError("type(x1) and type(x2) must be numpy.ndarray")
    '''
    
    # If x2 was not provided, then compute the
    # Lorentz inner product of x1 times itself.
    if type(x2) == str: x2 = x1
    
    # If dim was not provided, then compute it
    if not dim: dim = x1.shape[0]-1
             
    
    if type(dim) != int:
        raise ValueError("`dim` must be an integer")
    elif x1.shape != x2.shape:
        raise ValueError("Number of dimensions of multiplied vectors must be equal.")
    elif (x1.shape[0] - 1) != dim:
        raise ValueError("`dim` must equal x1.shape[0] - 1")
    elif (x1.shape[0] != 3) and (x1.shape[0] != 4):
        raise ValueError("Must give input as a numpy.ndarray with shape (3,_) or (4,_)")
    
    # Create list of "lorentz coefficients" i.e. the coefficients
    # of the terms in the Lorentz IP. 
    # If 2d, should be [1, 1, -1]; if 3D, should be [1,1,1,-1]
    # TODO: just remove dim?
    lorentz_coeffs = np.array([1]*dim + [-1])
 
    # Compute the inner product with a three-vector "dot-product"
    return sum(a*b*c for a, b, c in zip(x1, x2, lorentz_coeffs))



def localize_sound(
    positions,
    times,
    temp,
    invert_alg = 'gps', #options: 'lstsq', 'gps'
    center = True, #True = original Sound Finder behavior
    pseudo = True #False = original Sound Finder
):
    
    '''
    Perform TDOA localization on a sound
    
    Localize a single sound using time delay of arrival
    equations described in the class handout ("Global Positioning
    Systems"). Localization can be performed in a global coordinate
    system in meters (i.e., UTM), or relative to recorder positions
    in meters.
    
    Inputs:
        positions: a pandas dataframe with columns x, y, z, 
          indexed by the recorder name. There should be one row
          for each recorder, and their order should exactly 
          match the order of the recorder columns in the `times` df.
          Positions should be in meters, e.g., the UTM coordinate system.
          
        times: a pandas dataframe with column headers matching the contents
          and order of the recorders in the `positions` dataframe. 
          The dataframe should have only one row (for a single sound).
          The times should be in seconds.
          
        temp: a single-row pandas series containing the float value
          of the temperature in Celsius at which the time was created.
          
        invert_alg: what inversion algorithm to use
        
        center: whether to center recorders before computing localization
          result. Computes localization relative to centered plot, then
          translates solution back to original recorder locations. 
          (For behavior of original Sound Finder, use True)
        
        pseudo: whether to use the pseudorange error (True) or 
          sum of squares discrepancy (False) to pick the solution to return
          (For behavior of original Sound Finder, use False. However, 
          in initial tests, pseudorange error appears to perform better.)
          
    Returns:
        The solution with the lower sum of squares discrepancy
    '''
    
    
    # The number of dimensions in which to perform localization
    # 1 is subtracted to avoid counting the index_col (recorder number)
    # as a dimension
    dim = positions.shape[1]

    # Calculate speed of sound
    speeds = 331.3 * np.sqrt(1 + temp / 273.15)
    
    ##### Center recorders #####
    if center:
        warnings.warn("centering")
        x_center = positions['x'].mean()
        y_center = positions['y'].mean()
        positions_centered = positions.copy()
        positions_centered['x'] = positions_centered['x'] - x_center
        positions_centered['y'] = positions_centered['y'] - y_center
        if dim == 3:
            z_center = positions['z'].mean()
            positions_centered['z'] = positions_centered['z'] - z_center
        #print('positions_centered',positions_centered)
        positions = positions_centered
    else:
        warnings.warn("not centering")

    
    ##### Compute B, a, and e #####
    # Find the pseudorange, rho, for each recorder
    rho = times.multiply(-1 * speeds, axis='rows')
    rho.rename(index = {0:'rho'}, inplace = True)
    
    # Concatenate the pseudorange column to form matrix B
    rho = rho.T
    B = pd.concat([positions, rho], axis=1)
    
    # Vector of ones
    e = pd.DataFrame([1] * positions.shape[0])
    
    # The vector of squared Lorentz norms
    a = pd.DataFrame(0.5 * B.apply(lorentz_ip, axis=1))
    

    #print('invert_alg', invert_alg)
        
    if invert_alg == 'lstsq':
        # Closest equivalent to R's solve(qr(B), e)
        Bplus_e = np.linalg.lstsq(B, e, rcond=None)[0]
        Bplus_a = np.linalg.lstsq(B, a, rcond=None)[0]

    
    else: # invert_alg == 'gps' or 'special'
        ## Compute B+ = (B^T \* B)^(-1) \* B^T
        # B^T * B
        
        to_invert = np.matmul(B.T, B)
        try:
            inverted = np.linalg.inv(to_invert)
            
        except np.linalg.LinAlgError as err: 
            # Simply fail
            if invert_alg == 'gps':
                warnings.warn('4')
                if 'Singular matrix' in str(err):
                    warnings.warn('5')
                    warnings.warn("Singular matrix. Were recorders linear or on same plane? Exiting with NaN outputs", UserWarning)
                    return [[np.nan]]*(dim+1)
                else:
                    warnings.warn('6')
                    raise
                     
            
            # Fall back to lstsq algorithm
            else: # invert_alg == 'special'
                warnings.warn('7')
                
                Bplus_e = np.linalg.lstsq(B, e, rcond=None)[0]
                Bplus_a = np.linalg.lstsq(B, a, rcond=None)[0]

    # The whole thing
    Bplus = np.matmul(inverted, B.T)

    # Compute B+ * a and B+ * e
    Bplus_a = np.matmul(Bplus, a)
    Bplus_e = np.matmul(Bplus, e)   



    ###### Solve quadratic equation for lambda #####
    
  
    # Compute coefficients
    cA = lorentz_ip(Bplus_e)
    cB = 2*(lorentz_ip(Bplus_e, Bplus_a) -1)
    cC = lorentz_ip(Bplus_a)
    
    # Compute discriminant
    disc = cB**2 - 4 * cA * cC
    # If discriminant is negative, set to zero to ensure
    # we get an answer, albeit not a very good one
    if disc < 0: 
        disc = 0
        warnings.warn("Discriminant negative--set to zero. Solution may be inaccurate. Inspect final value of output array", UserWarning)
    
    # Compute options for lambda
    lamb = (-cB + np.array([-1, 1])*np.sqrt(disc))/(2*cA)
    
    # Find solution 0 and solution 1
    ale0 = np.add(a, lamb[0] * e)
    u0 = np.matmul(Bplus, ale0)
    ale1 =  np.add(a, lamb[1] * e)
    u1 = np.matmul(Bplus, ale1)
    
    #print('Solution 1: {}'.format(u0))
    #print('Solution 2: {}'.format(u1))
    
    
    ##### Return the better solution #####
     
    # Re-translate points
    if center:
            u0[0] += x_center
            u0[1] += y_center
            if dim == 3:
                u0[2] += z_center 
                    
            u1[0] += x_center
            u1[1] += y_center
            if dim == 3:
                u1[2] += z_center

    # Select and return quadratic solution
    if pseudo:
        # Return the solution with the lower error in pseudorange 
        # (Error in pseudorange is the final value of the position/solution vector)
        if abs(u0[-1]) <= abs(u1[-1]): return u0
        else: return u1
        
    else:    
        # This was the return method used in the original Sound Finder,
        # but it gives worse performance
        
        # Compute sum of squares discrepancies for each solution
        s0 = float(np.sum((np.matmul(B, u0) - np.add(a, lamb[0] * e))**2))
        s1 = float(np.sum((np.matmul(B, u1) - np.add(a, lamb[1] * e))**2)) 

        # Return the solution with lower sum of squares discrepancy
        if s0 < s1: return u0 
        else: return u1
    
    
    


def dfs_from_files(position_filename, time_filename):
    '''
    Test dataframes created from input .CSVs
    
    Ensures that the input .CSVs were in the expected format, including:
      - the final column of `times`, the DF of time delays 
        and temperatures, is titled 'temp'
      - `positions` has exactly the same index, in exactly the same order, 
        as the rows of `times`, except for 'temp', the final column of times.
    
    Returns dataframes for positions, times, and temps
    '''
    
    # Create a dataframe for positions
    positions = pd.read_csv(position_filename, index_col='recorder').astype('float64')
    
    # Create a matrix for times
    times = pd.read_csv(time_filename).astype('float64')
    #times.index = pd.Int64Index(range(times.shape[0]), dtype='int64')
   
    assert list(times)[-1:] == ['temp'],\
        "the final column of the TDOA .csv must be 'temp'"
    assert list(positions.index.values) == list(times)[1:-1],\
        """the recorders listed in the positions .csv (rows) must be
        the same as the recorders in the TDOA .csv (columns)"""
    
    
    # Create a dataframe for positions
    positions = pd.read_csv(position_filename, index_col='recorder').astype('float64')

    # Create a dataframe for times
    times_full = pd.read_csv(time_filename, index_col='idx').astype('float64')
    # Change index type to str (aka dtype = 'pandas.indexes.base.Index') 
    # This is necessary so that the df acquired from times.loc[[]] 
    # doesn't change the dtype of the index
    times_full.index = times.index.astype(str)

    assert list(times_full)[-1:] == ['temp'],\
        "the final column of the TDOA .csv must be 'temp'"

    # Compare the recorders in the index of the positions
    # to the recorders in the columns of the times
    assert list(positions.index.values) == list(times_full)[:-1],\
        """the recorders listed in the positions .csv (rows) must be
        the same as the recorders in the TDOA .csv (columns)"""

    times = times_full.drop(labels = 'temp', axis=1)
    temps = times_full[['temp']]
   
    return positions, times, temps
    
    
def all_sounds(position_filename, time_filename, plot=False):  

    # Validate .csvs and return proper dataframes
    positions, times, temps = dfs_from_files(position_filename, time_filename)
    
    sound_locs = []
    
    # Localize all sounds
    for sound_id in list(times.index.values):
        # Get only one row of TDOAs to localize
        one_times_row = times.loc[[sound_id]]
        one_temp = temps.loc[[sound_id]].temp

        # Find recorders where time is null, i.e. sound did not arrive at recorder
        no_times = one_times_row.columns[one_times_row.isnull().any(0).nonzero()[0]]

        # Drop recorder columns for those recorders
        one_times_row.drop(no_times, axis='columns', inplace=True)
        heard_positions = positions.drop(no_times, axis='index')

        # Localize the sound
        solution = localize_sound(heard_positions, one_times_row, one_temp)
        sound_locs.append(solution)

        # Plot the solution 
        if plot: plot_solution(positions, solution)
    
    print(sound_locs)
    return(sound_locs)
