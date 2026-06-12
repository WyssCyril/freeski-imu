import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path

##########################################################################################

fs_forcePlate = 1000 # in Hz

##########################################################################################

# load forceplate data and prepare it for use
def load_data_forcePlate(filename, forcePlateNr=4):

    # load csv
    df = pd.read_csv(filename, sep=",", header=3, skiprows=[4])

    # keep values of specified force plate only
    i = forcePlateNr - 1
    df = df[[f"Fx.{i}", f"Fy.{i}", f"Fz.{i}"]].rename(columns={f"Fx.{i}": "Fx", f"Fy.{i}": "Fy", f"Fz.{i}": "Fz"})
    
    # make Fz positive
    df['Fz'] = - df['Fz'] 

    return df

##########################################################################################

# detect jumps on the forceplate
def detect_jumps_forcePlate(df, bodyweight, th_jump=20, fs=1000, min_duration_s=0.05):

    Fz = df['Fz']
    
    # detect core of jump phase
    jump_mask = Fz < th_jump
    if not np.any(jump_mask):
        raise ValueError("No core flight segment (near-zero force) found")

    # detect transitions
    transitions = jump_mask.astype(int).diff()
    jump_starts = Fz.index[transitions == 1].tolist()
    jump_ends = Fz.index[transitions == -1].tolist()

    # handle cases where jump does not start on the plate -> i.e. only landing
    if jump_mask.iloc[0]:
        jump_starts.insert(0, df.index[0])
    
    jumps = []

    # pair start and end to detect jump segments
    for start_idx in jump_starts:
        # find the next landing
        end_idx = next((e for e in jump_ends if e > start_idx), None)
        if end_idx is None:
            continue

        # remove jump sequences that are too short
        if end_idx < (start_idx + (fs * min_duration_s)): 
            continue

        # take first sample after takeoff (Fz < bodyweight)
        takeoff_idx = next((i for i in range(start_idx, -1, -1) if Fz[i] >= bodyweight), 0)
        if takeoff_idx != 0:
            takeoff_idx = takeoff_idx + 1
            
        # take last sample before landing (Fz >= bodyweight)
        landing_idx = next((i for i in range(end_idx, len(Fz)) if Fz[i] >= bodyweight), -1)
        if landing_idx != -1:
            landing_idx = landing_idx - 1

        # detect peak after landing
        offset = int(fs / 4)
        Fpeak_idx = Fz[landing_idx:landing_idx+offset].idxmax()
        Fpeak = Fz[Fpeak_idx]

        # compute net force and relative net force
        Fpeak_net = Fpeak - bodyweight
        Fpeak_net_rel = Fpeak_net / bodyweight

        # compute RFD from landing to peak -> delta_F / delta_t
        delta_t = (Fpeak_idx - landing_idx) / fs
        RFD_to_peak = (Fpeak - Fz[landing_idx]) / delta_t
        RFD_to_peak_rel = RFD_to_peak / bodyweight

        # compute impulse from landing to peak, i.e. the area under the curve
        Impulse = np.trapz(Fz[landing_idx:Fpeak_idx+1], dx=1/fs)
        Impulse_net = np.trapz(Fz[landing_idx:Fpeak_idx+1] - bodyweight, dx=1/fs)
        Impulse_net_rel = Impulse_net / bodyweight

        # compute flight time
        if takeoff_idx != 0:
            t_flight_s = (landing_idx - takeoff_idx) / fs
        else:
            t_flight_s = 0
        
        jumps.append({
            'takeoff_idx': takeoff_idx,
            'landing_idx': landing_idx,
            'flight_time_s': t_flight_s,
            'time_to_peak': delta_t,
            'Fpeak_idx': Fpeak_idx,
            'Fpeak_N': Fpeak,
            'Fpeak_net_N': Fpeak_net,
            'Fpeak_net_normBW': Fpeak_net_rel,
            'RFD_to_peak': RFD_to_peak,
            'RFD_to_peak_normBW': RFD_to_peak_rel,
            'Impulse_to_peak_Ns': Impulse,
            'Impulse_to_peak_net_Ns': Impulse_net,
            'Impulse_to_peak_net_normBW': Impulse_net_rel
        })

    return pd.DataFrame(jumps)

##########################################################################################

# plot detected jumps on force plate
def plot_jumps_forcePlate(df, df_jumps, bodyweight, series_name, folder="images/", saveFig=False, show=True):

    fig = plt.figure(figsize=(10, 4))

    # plot force
    plt.plot(df["Fz"], label='Fz')

    # threshold line
    plt.axhline(bodyweight, color='gray', linestyle='--', linewidth=0.8,  label='Bodyweight')
    
    # takeoff & landing markers
    for i, row in df_jumps.iterrows():
        if row['takeoff_idx'] != 0:
            plt.axvline(row['takeoff_idx'], color='g', linestyle='--', linewidth=0.8, label='Takeoff' if i == 0 else "")
        plt.axvline(row['landing_idx'], color='r', linestyle='--', linewidth=0.8, label='Landing' if i == 0 else "")
        plt.axvline(row['Fpeak_idx'], color='darkorange', linestyle='--', linewidth=0.8, label='Peak' if i == 0 else "")

        # fill area under curve from landing → peak
        idx_range = np.arange(row['landing_idx'], row['Fpeak_idx'] + 1)
        plt.fill_between(idx_range, df["Fz"][idx_range], color='bisque', alpha=0.3, label='Impulse to peak' if i == 0 else "")

    # set x-limits to capture interesting part of the jump
    if not df_jumps.empty:
        margin = int(fs_forcePlate * 0.75)
        # check if first takeoff_idx is > 0
        if df_jumps['takeoff_idx'].min():
            x_min = max(0, df_jumps['takeoff_idx'].min() - margin)
        else:
            x_min = max(0, df_jumps['Fpeak_idx'].min() - margin)
        x_max = min(len(df) - 1, df_jumps['Fpeak_idx'].max() + margin)
        plt.xlim(x_min, x_max)

    # labels and formatting
    plot_name = f"{series_name}_forcePlate"
    plt.title(f"Detected Jumps - {plot_name}")
    plt.xlabel("Sample")
    plt.ylabel("Fz [N]")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    if saveFig:
        image_path = folder / f"{plot_name}.png"
        plt.savefig(image_path, dpi=300)
        print(f"Plot saved as {plot_name}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return