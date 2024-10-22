import os
import sys
import numpy as np
import pandas as pd

# Import the encoding to work with ezpadova
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
os.system('chcp 65001')

#Load the configuration file to make aviable the configuration variable
try:
    from ezpadova.config import reload_configuration

    reload_configuration()
except:
    pass

#Import the class and the variables to load the isochrones data from CMB website using ezpadova library
from ezpadova.parsec import get_isochrones
from ezpadova.config import configuration


def load_data(min_age=0.0,max_age=0.0,age_step=0.0,Z_min=0.0152,Z_max=0.03,Z_step=0.0,phot='YBC_gaiaEDR3'):
    #Get the phat to the correct file
    print(configuration.keys())
    photometric_system=configuration['photsys_file'][phot][1]
    
    if min_age==0.0 and max_age==0.0:
        df=get_isochrones(default_ranges=True,photsys_file=photometric_system)
    else:
        df=get_isochrones(age_yr=(min_age,max_age,age_step),Z=(Z_min,Z_max,Z_step),photsys_file=photometric_system)
    
    #Groupping the data as function of logAge and initial metallicity
    groups=df.groupby(['logAge','Zini'])
    
    df_out= dict(list(groups))
    return df_out

######################Simple population functions ######################
#these functions work only for a df that rappresents a simple population#
#with a single value of Zini and logAge.                                #
#########################################################################

def calculate_IMF(df, mini_column='Mini', imf_column='int_IMF'):
    """
    Calculates the IMF (Initial Mass Function) from a given DataFrame.

    Parameters:
    - df: DataFrame containing the 'Mini' and 'int_IMF' columns.
    - mini_column: Name of the column containing the 'Mini' values (default is 'Mini').
    - imf_column: Name of the column containing the 'int_IMF' values (default is 'int_IMF').

    Returns:
    - IMF: Array of interpolated IMF values.
    """
    # Ensure 'Mini' and 'int_IMF' are present in the DataFrame
    if mini_column not in df.columns or imf_column not in df.columns:
        raise ValueError(f"Columns '{mini_column}' and/or '{imf_column}' are not in the DataFrame.")
    
    # Extract the Mini (x-values) and int_IMF (integral) arrays
    Mini = df[mini_column].values  # Array of x
    integral_IMF = df[imf_column].values  # Array of the integral of Y(x)

    # Compute the difference of the integral (Y values)
    d_integral_IMF = np.diff(integral_IMF)  # Successive differences of the integral
    d_Mini = np.diff(Mini)  # Successive differences of Mini (x values)
    
    # Handle division by zero: Replace zeros with a small value (e.g., 1e-10)
    d_Mini[d_Mini == 0] = 1e-10

    # Compute Y(x)
    Y = d_integral_IMF / d_Mini

    # Append the last Y value to ensure the IMF array length matches the original data
    IMF = np.append(Y, Y[-1])

    return IMF

def remove_non_strictly_decreasing(df, imf_array, mini_column='Mini'):
    """
    Removes rows where the IMF array is not strictly decreasing with respect to the 'Mini' column in the DataFrame.

    Parameters:
    - df: The original DataFrame.
    - mini_column: The name of the 'Mini' column.
    - imf_array: The external IMF array that should be strictly decreasing.

    Returns:
    - A DataFrame with rows removed where the IMF array is not strictly decreasing.
    """
    # Sort the DataFrame by the 'Mini' column
    df_sorted = df.sort_values(by=mini_column).reset_index(drop=True)

    # Use the IMF array as-is without sorting (it corresponds directly to the Mini column)
    imf_sorted = imf_array

    # Find rows where the IMF array is not strictly decreasing
    mask_strictly_decreasing = pd.Series(imf_sorted).diff().fillna(0) < 0

    # Apply the mask to the sorted DataFrame
    df_filtered = df_sorted[mask_strictly_decreasing.values]

    return df_filtered

def interpolate_with_equispaced_mini(df, mini_column, num_points):
    """
    Adds equispaced points to the 'Mini' column of a DataFrame and applies interpolation
    to all other columns.

    Parameters:
    - df: the original DataFrame
    - mini_column: the name of the column containing the values to be made equispaced (e.g., "Mini")
    - num_points: the number of equispaced points to generate for the 'Mini' column

    Returns:
    - A new DataFrame with equispaced 'Mini' points and interpolated values.
    """

    # Step 1: Create new equispaced values for the 'Mini' column
    mini_min = df[mini_column].min()
    mini_max = df[mini_column].max()
    new_mini = np.linspace(mini_min, mini_max, num=num_points)

    # Step 2: Create a DataFrame with the new 'Mini' points
    df_new_mini = pd.DataFrame({mini_column: new_mini})

    # Step 3: Merge the two DataFrames on the 'Mini' column with an outer join
    df_merged = pd.merge(df_new_mini, df, on=mini_column, how='outer').sort_values(mini_column)

    # Step 4: Interpolate all numeric columns
    df_interpolated = df_merged.interpolate(method='linear')

    return df_interpolated

def generate_sintetic_diagram_from_isochrone(df, N_samples=10000):
    # Rimuovere duplicati dalla colonna int_IMF
    df_unique = df.drop_duplicates(subset='int_IMF')
    
    
    int_IMF = df_unique['int_IMF']
    
    # Genera numeri casuali uniformi
    uniform = np.random.uniform(np.min(int_IMF), np.max(int_IMF), N_samples)
    
    
    # Trova indici dove int_IMF >= uniform
    indices = np.searchsorted(int_IMF, uniform, side='right')
    
    # Gestisci eventuali indici fuori dal range
    indices = np.clip(indices, 0, len(df_unique) - 1)  # Usa df_unique qui
    
    df_out = df_unique.iloc[indices].copy()  # Copia delle righe selezionate dal DataFrame unico
    return df_out

def generate_cleened_sintetic_diagram_from_isochrone(r,N_samples=1e6):
    
    #Interpolation of the df to better rappresent the IMF with the generated data
    r_interpolate=interpolate_with_equispaced_mini(r,'Mini',int(1e3))
    
    #Evaluation of the IMF for each single value
    IMF_interpolate=calculate_IMF(r_interpolate)
    
    #Elimitation of non decrising IMF to ensure a right generation of the star population
    r_interpolate=remove_non_strictly_decreasing(r_interpolate,IMF_interpolate)
    
    #Re evaluation of the IMF
    IMF_interpolate=calculate_IMF(r_interpolate)
    
    #Generation of the syntetic diagram's values
    df_out_int= generate_sintetic_diagram_from_isochrone(r_interpolate, N_samples=int(N_samples))
    return df_out_int


######################functions for multiple values ####################
#                            of logAge and Zini                        #
########################################################################

def generate_syntetic_diagrams(df,N_samples=1e4):
    df_out={}
    for (logage,zini), df in  df.items():
        df_out[(logage,zini)]= generate_cleened_sintetic_diagram_from_isochrone(df,N_samples)
        
    return df_out

print('training classes imported')
