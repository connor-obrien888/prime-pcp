import numpy as np
import pandas as pd
import xarray as xr

def omni_loader(trange, fill_val = None, omni_varnames = ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'Vx', 'Vy', 'Vz', 'proton_density', 'Pressure', 'Mach_num', 'Mgs_mach_num']):
    """
    Load OMNI data from CDAWeb using cdasws package
    Parameters:
    trange : list of str
        A list of two strings representing the start and end of the time range
        for the OMNI data. The format of the strings should be 'YYYY-MM-DD HH:MM:SS+0000'
    fill_val : float, optional
        The value to fill in for missing data. The default is None.
    omni_varnames : list of str, optional
        The list of OMNI variable names to load. The default is ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'Vx', 'Vy', 'Vz', 'proton_density', 'Pressure', 'Mach_num', 'Mgs_mach_num'].
    Returns:
    omni_df : pandas DataFrame
        A DataFrame containing the OMNI data in the format used by the Mshpy package
    """
    from cdasws import CdasWs
    cdas = CdasWs()
    omni_vars = cdas.get_data('OMNI_HRO_1MIN', omni_varnames, trange[0], trange[1])
    #Make a dataframe iin the Mshpy input format
    omni_df = pd.DataFrame(omni_vars[1]['Epoch'].dt.year, columns=['Year'])
    omni_df['DOY'] = omni_vars[1]['Epoch'].dt.dayofyear
    omni_df['HR'] = omni_vars[1]['Epoch'].dt.hour
    omni_df['MN'] = omni_vars[1]['Epoch'].dt.minute
    if 'BX_GSE' in omni_varnames:
        omni_df['Bx'] = omni_vars[1]['BX_GSE']
    else:
        omni_df['Bx'] = fill_val
    if 'BY_GSE' in omni_varnames:
        omni_df['By'] = omni_vars[1]['BY_GSE']
    else:
        omni_df['By'] = fill_val
    if 'BZ_GSE' in omni_varnames:
        omni_df['Bz'] = omni_vars[1]['BZ_GSE']
    else:
        omni_df['Bz'] = fill_val
    if 'Vx' in omni_varnames:
        omni_df['Vx'] = omni_vars[1]['Vx']
    else:
        omni_df['Vx'] = fill_val
    if 'Vy' in omni_varnames:
        omni_df['Vy'] = omni_vars[1]['Vy']
    else:
        omni_df['Vy'] = fill_val
    if 'Vz' in omni_varnames:
        omni_df['Vz'] = omni_vars[1]['Vz']
    else:
        omni_df['Vz'] = fill_val
    if 'proton_density' in omni_varnames:
        omni_df['n'] = omni_vars[1]['proton_density']
    else:
        omni_df['n'] = fill_val
    if 'Pressure' in omni_varnames:
        omni_df['Pd'] = omni_vars[1]['Pressure']
    else:
        omni_df['Pd'] = fill_val
    if 'Mach_num' in omni_varnames:
        omni_df['Ma'] = omni_vars[1]['Mach_num']
    else:
        omni_df['Ma'] = fill_val
    if 'Mgs_mach_num' in omni_varnames:
        omni_df['Mm'] = omni_vars[1]['Mgs_mach_num']
    else:
        omni_df['Mm'] = fill_val
    #Fill nan values with fill_val if specified
    if fill_val is not None:
        omni_df = omni_df.fillna(fill_val)
    return omni_df

def omni_saver(trange, sep = ' ', omni_varnames = ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'Vx', 'Vy', 'Vz', 'proton_density', 'Pressure', 'Mach_num', 'Mgs_mach_num']):
    '''
    Save OMNI data from CDAWeb using cdasws package
    Parameters:
    trange : list of str
        A list of two strings representing the start and end of the time range
        for the OMNI data. The format of the strings should be 'YYYY-MM-DD HH:MM:SS+0000'
    sep : str, optional
        The separator to use when saving the data to a file. The default is ' '.
    omni_varnames : list of str, optional
        The list of OMNI variable names to load. The default is ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'Vx', 'Vy', 'Vz', 'proton_density', 'Pressure', 'Mach_num', 'Mgs_mach_num'].
    Returns:
    saved : bool
        A boolean indicating whether the data was successfully saved
    '''
    omni_df = omni_loader(trange, omni_varnames = omni_varnames) #Load the OMNI data
    #Save the data to a file
    saved = omni_df.to_csv('omni.lst', sep = sep, index = False, header = False)
    return saved

def mms_loader(trange, fill_val = None, mms_varnames = ['mms1_mec_r_gse']):
    """
    Load MMS data from CDAWeb using cdasws package
    Parameters:
    trange : list of str
        A list of two strings representing the start and end of the time range
        for the MMS data. The format of the strings should be 'YYYY-MM-DD HH:MM:SS+0000'
    fill_val : float, optional
        The value to fill in for missing data. The default is None.
    mms_varnames : list of str, optional
        The list of MMS variable names to load. The default is ['B_GSE', 'V_GSE', 'N', 'P', 'Mach_num', 'Mgs_mach_num'].
    Returns:
    mms_df : pandas DataFrame
        A DataFrame containing the MMS data in the format used by the Mshpy package
    """
    from cdasws import CdasWs
    cdas = CdasWs()
    mms_vars = cdas.get_data('MMS1_MEC_SRVY_L2_EPHT89D', mms_varnames, trange[0], trange[1])
    #Make a dataframe in the Mshpy input format
    mms_df = pd.DataFrame(mms_vars[1]['Epoch'].dt.strftime('%y/%m/%d'), columns=['Date'])
    mms_df['Time'] = mms_vars[1]['Epoch'].dt.strftime('%H:%M:%S')
    if 'mms1_mec_r_gse' in mms_varnames:
        mms_df['X'] = mms_vars[1]['mms1_mec_r_gse'][:, 0]
        mms_df['Y'] = mms_vars[1]['mms1_mec_r_gse'][:, 1]
        mms_df['Z'] = mms_vars[1]['mms1_mec_r_gse'][:, 2]
    return mms_df

def mms_saver(trange, sep = ' ', mms_varnames = ['mms1_mec_r_gse']):
    '''
    Save MMS data from CDAWeb using cdasws package
    Parameters:
    trange : list of str
        A list of two strings representing the start and end of the time range
        for the MMS data. The format of the strings should be 'YYYY-MM-DD HH:MM:SS+0000'
    sep : str, optional
        The separator to use when saving the data to a file. The default is ' '.
    mms_varnames : list of str, optional
        The list of MMS variable names to load. The default is ['mms1_mec_r_gse'].
    Returns:
    saved : bool
        A boolean indicating whether the data was successfully saved
    '''
    mms_df = mms_loader(trange, mms_varnames = mms_varnames) #Load the MMS data
    #Save the data to a file
    saved = mms_df.to_csv('orbit.txt', sep = sep, index = False, header = False)
    return saved

def generate_samples(slice, n):
    """
    Generate samples from a slice of PRIME outputs
    Parameters:
    slice : xarray Dataset
        The slice from which to generate samples using PRIME's mean and std
    n : int
        The number of samples to generate
    Returns:
    samples : xarray Dataset
        An array of samples generated from the slice using PRIME's mean and std
    """
    #Create a dictionary to store the samples
    samples = {}
    for key in slice.keys():
        #If the key ends in '_sig', is PC data, or is interp_frac skip it
        if key.endswith('_sig')|key.startswith('PC')|key.startswith('interp_frac'):
            continue
        #Generate samples using the mean and std
        samples[key] = xr.DataArray([np.random.normal(slice[key], slice[key + '_sig'], n)], coords={'Epoch':np.atleast_1d(slice['Epoch']), 'sample':np.arange(n)})
    #Convert the samples dictionary to a Dataset
    samples = xr.Dataset(samples)
    return samples

def read_cnvmap(filename):
    '''
    Read a SuperDARN .cnvmap file and return a list of dictionaries

    Parameters
    ----------
    filename : str
        The path to the .cnvmap file

    Returns
    -------
    cnvmap : list
        A list of dictionaries, each containing the data for a single time slice
    '''
    import pydarnio
    reader = pydarnio.SDarnRead(filename)
    cnvmap = reader.read_map()
    return cnvmap

def cnvmap_to_df(cnvmap):
    '''
    Convert a list of dictionaries (As read by read_cnvmap) to a pandas DataFrame

    Parameters
    ----------
    cnvmap : list
        A list of dictionaries, each containing the data for a single time slice

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the data from all time slices
    '''
    df = pd.DataFrame()
    times = extract_time(cnvmap)
    for i, slice in enumerate(cnvmap):
        slice_df = pd.DataFrame()
        for key in slice.keys():
            if (type(slice[key]) == np.ndarray): #If the value is an array figure out the length of the array and create a column for each element
                if key == 'nvec': #Special case for nvec, which we only need the sum of
                    slice_df[key] = np.sum(slice[key])
                elif key == 'vector.mlon': #Special case for the magnetic longitude of the vectors, which we need the range of
                    slice_df['lon_coverage'] = slice[key].max() - slice[key].min()
                else:
                    continue #This is currently disabled because it is too slow
                # for j in range(len(slice[key])): #Make a new Series for each key, then concatenate into slice_df (to avoid performance issues with appending to a DataFrame)
                #     temp = pd.Series(slice[key][j], name = key + str(j))
                #     slice_df = pd.concat([slice_df, temp], axis = 1)
            elif (key.startswith('start.')|key.startswith('end.')): #If the key starts with start. or end., it is time related and can be skipped
                continue
            else:
                slice_df[key] = [slice[key]]
        slice_df['time'] = times[i]
        df = pd.concat([df, slice_df])
    return df

def extract_time(cnvmap):
    '''
    Extract the time of each time slice from a list of dictionaries (As read by read_cnvmap)

    Parameters
    ----------
    cnvmap : list
        A list of dictionaries, each containing the data for a single time slice

    Returns
    -------
    times : list
        A list of datetime objects, each representing the time of a time slice
    '''
    times = [] #initialize empty list
    for slice in cnvmap:
        time = pd.to_datetime(str(slice['start.year']) + '-' + 
                              str(slice['start.month']).zfill(2) + '-' + 
                              str(slice['start.day']).zfill(2) + ' ' + 
                              str(slice['start.hour']).zfill(2) + ':' + 
                              str(slice['start.minute']).zfill(2) + ':' + 
                              str(slice['start.second']).zfill(2),
                              yearfirst=True,
                              utc = True)
        times.append(time)
    return times

def load_trange(start, stop, path = './data/', hemisphere = 'north', verbose = False):
    '''
    Load a range of .cnvmap files from time start to time stop

    Parameters
    ----------
    start : pandas.datetime
        The start time of the range (UTC)
    stop : pandas.datetime
        The end time of the range (UTC)
    path : str
        The path to the directory containing the yearly cnvmap folders (default is './data/')
    hemisphere : str
        The hemisphere to load data from (default is 'north')
    verbose : bool
        If True, print the time of each file as it is loaded (default is False)

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the data from all time slices in the range
    '''
    days = pd.date_range(start, stop, freq = 'D')
    df = pd.DataFrame()
    for day in days:
        try:
            cnvmap = read_cnvmap(path + str(day.year) + '_cnvmap/' + str(day.year) + str(day.month).zfill(2) + str(day.day).zfill(2) + '.' + hemisphere + '.cnvmap')
            df = pd.concat([df, cnvmap_to_df(cnvmap)])
            if verbose:
                print('Loaded ' + str(day))
        except:
            print('Error reading ' + str(day))
    df = df.reset_index(drop = True)
    return df

def data_load_routine(years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'], hemisphere = 'north', verbose = False):
    '''
    Load a range of .cnvmap files from a list of years, save into HDF, and combine into one dataframe.
    This is a script to show how the data was processed. It is not recommended to run this script as it will take a long time and use a lot of memory.
    Instead, use the pre-processed data in the data folder ('./data/superdarn.h5').

    Parameters
    ----------
    years : list
        A list of years to load data from (default is ['2013', '2014', '2015', '2016', '2017', '2018', '2019'])
    hemisphere : str
        The hemisphere to load data from (default is 'north')
    verbose : bool
        If True, print the time of each file as it is loaded (default is False)

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the data from all time slices in the range
    '''
    df = pd.DataFrame()
    for year in years:
        try:
            start = pd.to_datetime(year+'-01-01', utc=True)
            stop = pd.to_datetime(year+'-12-31', utc=True)
            superdarn = load_trange(start, stop)
            superdarn.to_hdf('./data/superdarn.h5', key = year+'_'+hemisphere, mode = 'a')
            if verbose:
                print('Loaded ' + year)
        except:
            print('Error reading ' + year)
        df = pd.concat([df, superdarn]) #Combine all years into one DataFrame
    df = df.drop_duplicates(subset='time') #Remove any duplicate rows
    df = df.reset_index(drop = True)
    df.to_hdf('./data/superdarn.h5', key = 'all_'+hemisphere, mode = 'a')
    return df

def EKL(vx, vy, vz, by, bz):
    '''
    Returns Kan-Lee electric field in mV/m given solar wind V and B.
    Could work for magnetosheath, but the Kan-Lee E field was not developed for the sheath so it is not recommended.

    Parameters
    ----------
    vx : float, array-like
        Solar wind GSE X velocity in km/s
    vy : float, array-like
        Solar wind GSE Y velocity in km/s
    vz : float, array-like
        Solar wind GSE Z velocity in km/s
    by : float, array-like
        Interplanetary magnetic field GSM Y component in nT
    bz : float, array-like
        Interplanetary magnetic field GSM Z component in nT

    Returns
    -------
    ekl : float, array-like
        Kan-Lee electric field in mV/m
    '''
    ekl = 0.001*np.sqrt(vx**2+vy**2+vz**2)*np.sqrt(by**2+bz**2)*(np.sin(np.arctan2(by, bz)/2))**2
    return ekl

def EKL_err(vx, vx_sig, vy, vy_sig, vz, vz_sig, by, by_sig, bz, bz_sig):
    '''
    Returns error in Kan-Lee electric field in mV/m given solar wind V and B and 1sigma uncertainties.
    Could work for magnetosheath, but the Kan-Lee E field was not developed for the sheath so it is not recommended.

    Parameters
    ----------
    vx : float, array-like
        Solar wind GSE X velocity in km/s
    vx_sig : float, array-like
        Solar wind GSE X velocity 1sigma error in km/s
    vy : float, array-like
        Solar wind GSE Y velocity in km/s
    vy_sig : float, array-like
        Solar wind GSE Y velocity 1sigma error in km/s
    vz : float, array-like
        Solar wind GSE Z velocity in km/s
    vz_sig : float, array-like
        Solar wind GSE Z velocity 1sigma error in km/s
    by : float, array-like
        Interplanetary magnetic field GSM Y component in nT
    by_sig : float, array-like
        Interplanetary magnetic field GSM Y component 1sigma error in nT
    bz : float, array-like
        Interplanetary magnetic field GSM Z component in nT
    bz_sig : float, array-like
        Interplanetary magnetic field GSM Z component 1sigma error in nT

    Returns
    -------
    ekl_err : float, array-like
        Kan-Lee electric field 1sigma error in mV/m
    '''
    dekl_dvx = 0.001*vx*((vx**2+vy**2+vz**2)**(-1/2))*np.sqrt(by**2+bz**2)*(np.sin(np.arctan2(by, bz)/2))**2
    dekl_dvy = 0.001*vy*((vx**2+vy**2+vz**2)**(-1/2))*np.sqrt(by**2+bz**2)*(np.sin(np.arctan2(by, bz)/2))**2
    dekl_dvz = 0.001*vz*((vx**2+vy**2+vz**2)**(-1/2))*np.sqrt(by**2+bz**2)*(np.sin(np.arctan2(by, bz)/2))**2
    dekl_dby = 0.001*np.sqrt(vx**2+vy**2+vz**2)*by*((by**2+bz**2)**(1/2))*(np.sin(np.arctan2(by, bz)/2))**2 + 0.001*np.sqrt(vx**2+vy**2+vz**2)*np.sqrt(by**2+bz**2)*by/(2*np.sqrt((by/bz)**2 + 1)*(by**2 + bz**2))
    dekl_dbz = 0.001*np.sqrt(vx**2+vy**2+vz**2)*bz*((by**2+bz**2)**(1/2))*(np.sin(np.arctan2(by, bz)/2))**2 - 0.001*np.sqrt(vx**2+vy**2+vz**2)*np.sqrt(by**2+bz**2)*(by**2)*bz*np.sqrt((by/bz)**2 + 1)/(2*(by**2 + bz**2)**2)

    ekl_err = np.sqrt((dekl_dvx*vx_sig)**2 + (dekl_dvy*vy_sig)**2 + (dekl_dvz*vz_sig)**2 + (dekl_dby*by_sig)**2 + (dekl_dbz*bz_sig)**2)
    return ekl_err

def EMR(vx, bz, by = None, rect = np.nan):
    '''
    Returns rectified Y component of motional electric field in a magnetized plasma.

    Parameters
    ----------
    vx : float, array-like
        Plasma X velocity in km/s
    bz : float, array-like
        Magnetic field Z component in nT
    by : float, array-like, optional
        Magnetic field Y component in nT. If specified, use the tangential magnetic field
        bt=sqrt(by^2 + bz^2) instead of bz. bt is still rectified for northward bz. Default None.
    rect : float, optional
        Value negative E fields are rectified to. Default NaN

    Returns
    -------
    emr : float, array-like
        Motional electric field Y component in mV/m
    '''
    if by is not None:
        emr = 0.001*np.abs(vx*np.where(bz <= 0, np.sqrt(bz**2 + by**2), rect))
    else:
        emr = 0.001*np.abs(vx*np.where(bz <= 0, bz, rect))
    return emr

def EMR_err(vx, vx_sig, bz, bz_sig, by = None, by_sig = None, rect = np.nan):
    '''
    Returns error in rectified Y component of motional electric field in a magnetized plasma.

    Parameters
    ----------
    vx : float, array-like
        Plasma X velocity in km/s
    vx_sig : float, array-like
        1sigma error in plasma X velocity in km/s
    bz : float, array-like
        Magnetic field Z component in nT
    bz_sig : float, array-like
        1sigma error in magnetic field Z component in nT
    by : float, array-like, optional
        Magnetic field Y component in nT. If specified, use the tangential magnetic field
        bt=sqrt(by^2 + bz^2) instead of bz AND require by_sig. bt is still rectified for northward bz. 
        Default None.
    by : float, array-like, optional
        1sigma error in magnetic field Y component in nT. If specified, use the tangential magnetic 
        field bt=sqrt(by^2 + bz^2) instead of bz AND require by. bt is still rectified for northward 
        bz. Default None.
    rect : float
        Value negative E fields are rectified to. Default NaN

    Returns
    -------
    emr_std : float, array-like
        1sigma error in motional electric field Y component in mV/m
    '''
    if (by is not None)&(by_sig is not None):
        bt = np.sqrt(by**2 + bz**2)
        bt_sig = (by+bz)/np.sqrt(by**2 + bz**2)
        emr_std = np.sqrt(
            (0.001*np.abs(vx_sig*np.where(bz <= 0, bt, rect)))**2 + #dEMR/dvx term
            (0.001*np.abs(vx*np.where(bz <= 0, bt_sig, rect)))**2 #dEMR/dbz term
        )
    else:
        emr_std = np.sqrt(
            (0.001*np.abs(vx_sig*np.where(bz <= 0, bz, rect)))**2 + #dEMR/dvx term
            (0.001*np.abs(vx*np.where(bz <= 0, bz_sig, rect)))**2 #dEMR/dbz term
        )
    return emr_std

def ERX(ni, vx, vy, vz, bx, by, bz, tiperp):
    '''
    Returns reconnection electric field in the sheath from Cassak-Shay equation in very asymmetric plasma limit.
    Not valid for solar wind. Becomes unreliable off the Sun-Earth line.

    Parameters
    ----------
    ni : float, array-like
        Magnetosheath ion density in cm^-3
    vx : float, array-like
        Magnetosheath GSE X velocity in km/s
    vy : float, array-like
        Magnetosheath GSE Y velocity in km/s
    vz : float, array-like
        Magnetosheath GSE Z velocity in km/s
    bx : float, array-like
        Magnetosheath magnetic field GSM Y component in nT
    by : float, array-like
        Magnetosheath magnetic field GSM Y component in nT
    bz : float, array-like
        Magnetosheath magnetic field GSM Z component in nT
    tiperp : float, array-like
        Magnetosheath perpendicular-to-B temperature in eV

    Returns
    -------
    erx : float, array-like
        Reconnection electric field in mV/m
    '''
    B = np.sqrt(bx**2+by**2+bz**2) #Magnetic field magnitude
    beta = 0.403*ni*tiperp/(B**2) #Plasma beta (perpendicular)
    erx = 0.02 * 0.2 * (B**2) * (((1+beta)/ni)**(1/2)) * (np.sin(np.arctan2(by, bz)/2))**2 #Reconnection electric field (modified Cassak-Shay)
    return erx

