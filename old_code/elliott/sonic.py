### sonic.py ###
# Elliott Walker #
# Last update: 19 July 2024 #
# Analysis of the snippet of sonic data #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datetime import datetime
import helper_functions as hf
import multiprocessing

WINDS = ['Ux','Uy','Uz'] # Columns containing wind speeds, in order
TEMPERATURES = ['Ts', 'amb_tmpr'] # Columns containing temperatures in C
IGNORE = ['H2O', 'CO2', 'Ts', 'amb_tmpr', 'amb_press'] # Columns we don't care about

class Logger:
    def __init__(self, logfile = 'output.log', pid = 0):
        self.is_printer = False
        self.is_void = False
        self.logfile = logfile
        self.pid = pid        
    
    def log(self, string, timestamp = False):
        log_string = f'[{datetime.now()}] {string}' if timestamp else str(string)
        pid = self.pid if self.pid else 'LOGPARENT'
        log_string = f'[[{pid}]] {log_string}'
        with open(self.logfile, 'a') as f:
            f.write(log_string+'\n')
        return

    def sublogger(self, pid = None):
        if pid is None:
            pid = os.getpid()
        self.log(f'Spawned sublogger for pid {pid}', timestamp=True)
        if self.is_printer:
            return Printer(pid = pid)
        if self.is_void:
            return VoidLogger()
        return Logger(logfile = self.logfile, pid = pid)
    
class VoidLogger(Logger):
    def __init__(self):
        Logger.__init__(self)
        self.is_void = True
    
    def log(self, string, timestamp = False):
        return
    
class Printer(Logger):
    def __init__(self, pid = 0):
        Logger.__init__(self, pid = pid)
        self.is_printer = True

    def log(self, string, timestamp = False):
        log_string = f'[{datetime.now()}] {string}' if timestamp else str(string) 
        if self.pid: log_string = f'[[{self.pid}]] {log_string}'
        print(log_string)
        return

# Loads dataframe: Handles timestamps, duplicate removal, column removal, and conversion.
def load_frame(filepath, # location of the CSV file to load
               kelvinconvert = TEMPERATURES, # columns which should be converted from C -> K
               ignore = IGNORE
               ):

    df = pd.read_csv(filepath, low_memory = False).rename(columns={'TIMESTAMP' : 'time'})

    df['time'] = pd.to_datetime(df['time'], format = 'mixed')
    df.set_index('time', inplace = True)
    df = df[~df.index.duplicated(keep = 'first')]
    df.sort_index(inplace = True)

    for col in df.columns:

        if col in ignore: # We don't care about 
            df.drop(columns = [col], inplace = True)
            continue

        df[col] = pd.to_numeric(df[col], errors = 'coerce')

        if col in kelvinconvert: # Any column listed in kelvinconvert will have its values converted from C to K
            df[col] += 273.15

    return df

# Compute autocorrelations. Returns a dataframe of autocorrelations, timestamped by lag length.
def compute_autocorrs(df, # Dataframe to work with
                     autocols = [], # Columns to compute autocorrelations for
                     maxlag = 0.5, # Work for lags from 0 up to <maxlag> * <(duration of df)>
                     verbose = False,
                     logger = None
                     ):
    
    if autocols == []: # If an empty list is passed in, use all columns
        autocols = df.columns.tolist()

    kept = int(len(df)*maxlag)
    lost = len(df) - kept

    df_autocorr = pd.DataFrame(df.copy().reset_index()['time'][:kept])

    for col in autocols:

        lag_range = range(kept)

        if logger:
            logger.log(f'Autocorrelating for {col}', timestamp = True)

        if verbose or (logger and logger.is_printer):
            lag_range = tqdm(lag_range)

        Raa = []
        for lag in lag_range:
            autocorr = df[col].autocorr(lag = lag)
            Raa.append(autocorr)
        df_autocorr[f'R_{col}'] = Raa

    df_autocorr.set_index('time', inplace = True)
    df_autocorr.sort_index(inplace = True)
    starttime = df_autocorr.index[0]
    deltatime = df_autocorr.index - starttime
    df_autocorr['lag'] = deltatime.days * 24 * 3600 + deltatime.seconds + deltatime.microseconds/1e6
    df_autocorr.reset_index(drop = True)
    df_autocorr.set_index('lag', inplace = True)

    if logger:
        logger.log(f'Completed autocorrelations', timestamp = True)

    return df_autocorr

# Generate autocorrelation plots, and either save them to <saveto> or show them
def plot_autocorrs(df_autocorr,
                   title = 'Autocorrelation Plot',
                   saveto = None,
                   threshold=0.):
    
    fig, ax = plt.subplots()
    fig.suptitle(title, fontweight = 'bold')

    ax.plot(df_autocorr.index, [threshold]*len(df_autocorr), c='tab:gray', label = f'threshold = {threshold}', linewidth=1)
    if threshold != 0.:
        ax.plot(df_autocorr.index, [0.]*len(df_autocorr), c='black', linestyle='dashed', linewidth=1)

    for col in df_autocorr.columns:
        ax.plot(df_autocorr.index, df_autocorr[col], label = str(col)[2:], linewidth = 1)

    ax.set_ylim(-0.2,1.1)
    ax.set_ylabel('Autocorrelation')
    ax.set_xlabel('Lag (s)')
    ax.legend()

    fig.tight_layout(pad = 1)

    if saveto is None:
        plt.show()
    else:
        plt.savefig(saveto, bbox_inches='tight')
    plt.close()

    return

# Compute integral time and length scales
def integral_scales(df, # dataframe containing original data
                    df_autocorr, # dataframe containing the autocorrelations as computed by compute_autocorrs
                    cols = [], # wind speed column names in <df>
                    threshold = 0.25, # integrate up to the first time that the autocorrelation dips below this threshold
                    logger = None # Logger object for output
                    ):

    scales = dict()

    if cols == []:
        collist = df_autocorr.columns.tolist()
        for col in collist:
            cols.append(col[2:])

    warn = False
    for col in cols:

        Raa = df_autocorr[f'R_{col}']
        dt = df_autocorr.index[1] - df_autocorr.index[0]

        mean = df[col].mean()

        cutoff_index = 0
        for i, val in enumerate(Raa):
            if val < threshold:
                cutoff_index = i
                break

        if cutoff_index == 0:
            warn = True
            if logger:
                logger.log(f'Warning: failed to find cutoff for integration (variable {col})')

        i_time = np.sum(Raa.loc[:cutoff_index]) * dt
        i_length = i_time * abs(mean)
        scales[col] = (i_time, i_length)

    return scales, warn

# Save information, including integral scales, to a text file
def save_scales(scales,
                filename,
                warn = False,
                ri = None,
                times = None,
                order = WINDS
                ):

    with open(filename, 'w') as f:

        if warn:
            f.write(f"Warning - at least one variable's autocorrelation did not fall below the threshold, possibly signifying nonstationary data.")

        if times:
            f.write(times+'\n')
            
        if ri:
            f.write(ri+'\n')

        vars = order if set(order) == set(scales.keys()) else scales.keys()
        for var in vars:
            i_time, i_length = scales[var]
            f.write(f'{var}: Time scale = {i_time:.3f} s, Length scale = {i_length:.3f} m\n')

    return

def match_ri(df, # dataframe which we want to match ri to, based on its start & end times
             df_ri, # dataframe containing ri values
             where = 'ri' # Ri column name
             ):

    start_time = df.index[0]
    end_time = df.index[-1]

    dfr = df_ri.reset_index()
    dfr['time'] = pd.to_datetime(dfr['time'])
    sliced = dfr[dfr['time'].between(start_time,end_time)]

    mean_ri = sliced[where].mean()
    median_ri = sliced[where].median()
    stability1 = hf.stability_class(mean_ri)
    stability2 = hf.stability_class(median_ri)
    stability = stability1 if stability1 == stability2 else f'{stability1}/{stability2}'

    ri_string = f'Bulk Ri: mean {mean_ri:.3f}, median {median_ri:.3f} ({stability})'

    return ri_string


def _analyze_file(args):
    filename, kelvinconvert, autocols, maxlag, threshold, savedir, df_match, savecopy, plotautocorrs, saveautocorrs, savescales, logparent, multiproc = args

    if multiproc:
        logger = logparent.sublogger()
    else:
        logger = logparent

    path = os.path.abspath(os.path.join(parent, filename))
    if not(os.path.isfile(path) and filename[-4:] == '.csv'):
        return
    logger.log(f'Loading {path}', timestamp = True)

    name = filename[:-4]
    intermediate = f'{savedir}/{name}'
    os.makedirs(intermediate, exist_ok = True)
    
    df = load_frame(path, kelvinconvert = kelvinconvert)

    if savecopy:
        fname = os.path.abspath(os.path.join(intermediate,'data.csv'))
        df.to_csv(fname)
        logger.log(f'Copied data to {fname}')

    starttime = df.index[0]
    endtime = df.index[-1]
    time_string = f'Time interval: {starttime} to {endtime}'
    logger.log(time_string)

    ri_string = None
    if df_match is not None:
        ri_string = match_ri(df, df_match)
        logger.log(ri_string)

    df_autocorr = compute_autocorrs(df, autocols = autocols, maxlag = maxlag, logger = logger)

    if saveautocorrs:
        fname = os.path.abspath(os.path.join(intermediate,'autocorrs.csv'))
        df_autocorr.to_csv(fname)
        logger.log(f'Saved autocorrelations to {fname}')

    if plotautocorrs:
        fname = os.path.abspath(os.path.join(intermediate,'autocorr.png'))
        plot_autocorrs(df_autocorr, title = f'{name} Autocorrelations', saveto = fname, threshold=threshold)
        logger.log(f'Saved plots to {fname}')

    if savescales:
        scales, warn = integral_scales(df, df_autocorr, cols = list(set(WINDS)&set(autocols)), threshold = threshold)
        fname = os.path.abspath(os.path.join(intermediate,'integralscales.txt'))
        save_scales(scales, filename = fname, warn = warn, ri = ri_string, times = time_string)
        logger.log(f'Saved info to {fname}')

    for var, s in scales.items():
        logger.log(f'Mean {var} = {df[var].mean():.3f} m/s')
        i_time, i_length = s
        logger.log(f'\tIntegral time scale = {i_time:.3f} s')
        logger.log(f'\tIntegral length scale = {i_length:.3f} m')


def analyze_directory(parent, 
                      *,
                      kelvinconvert = TEMPERATURES,
                      autocols = WINDS,
                      maxlag = 0.5,
                      threshold = 0.25,
                      savedir = '.',
                      matchfile = None,
                      savecopy = True,
                      plotautocorrs = True,
                      saveautocorrs = True,
                      savescales = True,
                      logger = Printer(),
                      nproc = 1
                      ):
    
    logger.log(f'Beginning analysis of {parent}', timestamp = True)
    if type(nproc) is int and nproc > 1:
        logger.log(f'MULTIPROCESSING ENABLED: {nproc=}')
        multiproc = True
    else:
        logger.log('Multiprocessing DISABLED.')
        nproc = 1
        multiproc = False
    
    if matchfile:
        df_match = pd.read_csv(matchfile)
        df_match.set_index('time', inplace = True)
    else:
        df_match = None

    arguments = (kelvinconvert, autocols, maxlag, threshold, savedir, df_match, savecopy, plotautocorrs, saveautocorrs, savescales, logger, multiproc)
    directory = [(filename, *arguments) for filename in os.listdir(parent)]

    pool = multiprocessing.Pool(processes = nproc)

    pool.map(_analyze_file, directory)

    pool.close()
    pool.join()

    logger.log(f'COMPLETED!', timestamp = True)

    return

def _confirm(message):
    response = input(message)
    if response.lower() == 'y':
        return True
    return False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        prog = 'sonic.py',
        description = 'Analyzes chunks of sonic data',
    )

    parser.add_argument('-c', '--clear', action = 'store_true', help = 'clear the target directory?')
    parser.add_argument('-y', '--yes', action = 'store_true', help = 'do not confirm before clearing?')
    parser.add_argument('-d', '--data', default = '../../DATA/KCC_106m_Flux_Tower_Data', help = 'input data directory')
    parser.add_argument('-t', '--target', default = 'sonic_results',  help = 'output target directory')
    parser.add_argument('-m', '--match', default = 'ten_minutes_labeled.csv', help = 'file containing bulk Ri to match')
    parser.add_argument('--nomatch', action = 'store_true', help = 'do not perform Ri match?')
    parser.add_argument('-n', '--nproc', default = 1, help = 'number of CPUs to run; sets verbose to False')
    parser.add_argument('-s', '--silent', action = 'store_true', help = 'neither print nor log?')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action = 'store_true', help = 'print to standard output instead of logging?')
    group.add_argument('-l', '--logfile', help = 'file to log to')

    args = parser.parse_args()

    nomatch = args.nomatch
    parent = args.data
    savedir = args.target
    matchfile = args.match

    if '/' not in savedir:
        savedir = f'./{savedir}'
    if '/' not in parent:
        parent = f'./{parent}'
    if '/' not in matchfile:
        matchfile = f'./{matchfile}'

    savedir = os.path.abspath(savedir)
    parent = os.path.abspath(parent)
    matchfile = os.path.abspath(matchfile)

    if not os.path.exists(parent):
        raise OSError(f'Data directory {parent} not found, exiting.')
    if not os.path.exists(matchfile):
        nomatch = True

    verbose = args.verbose
    if int(args.nproc) > 1 or args.silent: verbose = False

    if nomatch:
        matchfile = None

    if args.clear:
        if os.path.exists(savedir):
            if args.yes or _confirm(f'Really delete contents of {savedir}? (y/n): '):
                from shutil import rmtree
                rmtree(savedir)

    os.makedirs(savedir, exist_ok = True)

    if verbose:
        logger = Printer()
    elif args.silent:
        logger = VoidLogger()
    else:
        logfile = os.path.join(savedir, 'sonic_analysis.log')
        if args.logfile:
            logfile = args.logfile
            if '.' not in logfile:
                logfile += '.log'
            if '/' not in logfile:
                logfile = f'./{logfile}'
        logfile = os.path.abspath(logfile)
        logger = Logger(logfile = logfile)

    analyze_directory(parent,
                      maxlag = 0.5,
                      threshold = 0.5,
                      matchfile = matchfile,
                      savedir = savedir,
                      logger = logger,
                      nproc = int(args.nproc)
                      )
    