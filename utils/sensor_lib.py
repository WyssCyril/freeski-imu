import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from pathlib import Path
from datetime import datetime

########################################################################
#          Constants     
########################################################################
SAMPLE_RATE_IMU = 200 # in Hz
SAMPLE_RATE_GNSS = 10 # in Hz
BASE_PATH = Path("../Measurements/Messungen_Schnee/")

########################################################################
#         1. Data Import     
########################################################################

# --- load all files for one athlete on a given day ---
def load_measurements(date, place, athlete_id):

    folder = BASE_PATH / f"{date}_{place}/"
    pattern = f"{date}_{place}_{athlete_id}_*.csv"
    files = list(folder.glob(pattern))
    
    if not files:
        print(f"No files found for athlete {athlete_id} on {date} at {place}.")
        return {}
    
    dfs = {}
    for f in files:
        # extract sensor name and sensor type
        sensor_name = "_".join(f.stem.split("_")[3:4])
        sensor_type = "_".join(f.stem.split("_")[4:])

        # create key
        key = (date, place, f"Athlete{athlete_id}", sensor_name, sensor_type)
        
        dfs[key] = pd.read_csv(f)

    print(f"Athlete {athlete_id}: {len(dfs)} sensor file(s) loaded")
    
    return dfs


def load_runs(date, location, athlete_id):

    folder = BASE_PATH / f"{date}_{location}" / f"Athlete_{athlete_id}" / "runs"
    pattern = f"*.csv"
    files = list(folder.glob(pattern))
    if not files:
        print(f"No files found for athlete {athlete_id}.")
        return {}

    data = {}
    for f in files:
        # Extract sensor name
        run_name = "_".join(f.stem.split("_")[0:])
        data[run_name] = pd.read_csv(f)

    print(f"Athlete {athlete_id}: {len(data)} sensor file(s) loaded")
    
    return data
    

def create_subfolder_structure(folder):

    # folder for all 
    folder_analysis = folder / "analysis"
    folder_analysis.mkdir(parents=True, exist_ok=True)

    # folders for data storage
    folder_data = folder / "data"
    folder_data.mkdir(parents=True, exist_ok=True)

    folder_data_sessions = folder_data / "sessions"
    folder_data_sessions.mkdir(parents=True, exist_ok=True)
    
    folder_data_runs = folder_data / "runs"    
    folder_data_runs.mkdir(parents=True, exist_ok=True)

    folder_data_jumps = folder_data / "jumps"    
    folder_data_jumps.mkdir(parents=True, exist_ok=True)
    
    # folder for images
    folder_images = folder / "images"
    folder_images.mkdir(parents=True, exist_ok=True)
    
    folder_images_sessions = folder_images / "sessions"
    folder_images_sessions.mkdir(parents=True, exist_ok=True)
    
    folder_images_runs = folder_images / "runs"
    folder_images_runs.mkdir(parents=True, exist_ok=True)
    
    folder_images_jumps = folder_images / "jumps"
    folder_images_jumps.mkdir(parents=True, exist_ok=True)

    folders = {
        "root": folder,
        "analysis": folder_analysis,
        "data": {
            "root": folder_data,
            "sessions": folder_data_sessions,
            "runs": folder_data_runs,
            "jump_level": folder_data_jumps
        },
        "images": {
            "root": folder_images,
            "sessions": folder_images_sessions,
            "runs": folder_images_runs,
            "jumps": folder_images_jumps
        }
    }
        
    return folders


########################################################################
#         2. Data Preprocessing   
########################################################################   

# mask invalid gnss data
def mask_invalid_gnss_data(values, time, rate_max):

    mask = np.zeros(len(values), dtype=bool)
    last_valid_idx = None
    
    for i in range(len(values)):
        if np.isnan(values[i]):
            mask[i] = True
            continue

        if last_valid_idx is None:
            last_valid_idx = i
            continue

        dt = time[i] - time[last_valid_idx]
        rate = abs(values[i] - values[last_valid_idx]) / dt

        # mask samples where change is too high
        if rate > rate_max:
            mask[i] = True
        else:
            last_valid_idx = i
            
    return mask

# clean gnss data
def clean_gnss_data(df, ds_max=50, dv_max=40, dh_max=40, safety_samples=50, interpolate=True):
    
    df = df.copy()

    # compute delta t between all samples
    time = df['timestamp [us]'] * 1e-6
    dt = np.diff(time, prepend=time[0]-1/SAMPLE_RATE_GNSS)

    ##################################################################
    # 1) Detect parts where sensor is not fully initialized or signal is bad
    
    # mask samples where sensor has no speed information
    zero_speed_mask = ((df['speedN [m/s]'] == 0) &
                       (df['speedE [m/s]'] == 0) &
                       (df['speedD [m/s]'] == 0))
    
    # add safety interval where data may still be wrong
    padded_mask = zero_speed_mask.rolling(window=2*safety_samples + 1, center=True, min_periods=1).max().astype(bool)
    # declare samples during all-zero speed periods as invalid
    df.loc[padded_mask, ['latitude [deg]', 'longitude [deg]', 'altitude [m]']] = np.nan
    
    ##################################################################
    # 2) Detect corrupted or impossible values (e.g. spikes) 
    
    # maximum possible distance per dt
    # ds_max = s_max * dt * 2
    # dv_max = dv_max * dt
    # dAlt_max = alt_diff_max * dt * 2

    # estimate mean latitude
    mean_lat = df['latitude [deg]'].mean()
    
    # lat/lon degree-per-meter constants
    lat_deg_per_m = 1.0 / 111320.0 # approximation
    long_deg_per_m = 1.0 / (111320.0 * math.cos(math.radians(mean_lat)))

    # change in location too high
    # mask_lat = (df['latitude [deg]'].diff().abs() > ds_max * lat_deg_per_m)
    # mask_long = (df['longitude [deg]'].diff().abs() > ds_max * long_deg_per_m)
    mask_lat = mask_invalid_gnss_data(df['latitude [deg]'], time, 2 * ds_max * lat_deg_per_m)
    mask_long = mask_invalid_gnss_data(df['longitude [deg]'], time, 2 * ds_max * long_deg_per_m)
    
    # change in velocity too high or velocity too high
    # mask_speedN = (df['speedN [m/s]'].diff().abs() > dv_max) | (df['speedN [m/s]'].abs() > v_max)
    # mask_speedE = (df['speedE [m/s]'].diff().abs() > dv_max) | (df['speedE [m/s]'].abs() > v_max)
    # mask_speedD = (df['speedD [m/s]'].diff().abs() > dv_max) | (df['speedD [m/s]'].abs() > v_max)
    mask_speedN = df['speedN [m/s]'] > 50
    mask_speedN |= mask_invalid_gnss_data(df['speedN [m/s]'], time, dv_max)
    mask_speedE = df['speedE [m/s]'] > 50
    mask_speedE |= mask_invalid_gnss_data(df['speedE [m/s]'], time, dv_max)
    mask_speedD = df['speedD [m/s]'] > 50
    mask_speedD |= mask_invalid_gnss_data(df['speedD [m/s]'], time, dv_max)

    # altitude difference too high
    # mask_alt = (df['altitude [m]'].diff().abs() > dAlt_max) | (df['altitude [m]'] < 0)
    mask_alt = df['altitude [m]'] < 0
    mask_alt |= mask_invalid_gnss_data(df['altitude [m]'], time, dh_max * 2)

    # merge masks
    mask_position = mask_lat | mask_long | mask_alt
    mask_speed = mask_speedN | mask_speedE | mask_speedD
    mask_position = mask_position.rolling(window=5, center=True, min_periods=1).max().astype(bool)
    mask_speed = mask_speed.rolling(window=5, center=True, min_periods=1).max().astype(bool)

    # remove values in respective columns
    # cols_to_nan = ['latitude [deg]', 'longitude [deg]', 'speedN [m/s]', 'speedE [m/s]', 'speedD [m/s]', 'altitude [m]']
    cols_position = ['latitude [deg]', 'longitude [deg]', 'altitude [m]']
    cols_speed = ['speedN [m/s]', 'speedE [m/s]', 'speedD [m/s]']
    
    # df.loc[mask_lat, cols_to_nan[0]] = np.nan
    # df.loc[mask_long, cols_to_nan[1]] = np.nan
    # df.loc[mask_speedN, cols_to_nan[2]] = np.nan
    # df.loc[mask_speedE, cols_to_nan[3]] = np.nan
    # df.loc[mask_speedD, cols_to_nan[4]] = np.nan
    # df.loc[mask_alt, cols_to_nan[5]] = np.nan

    df.loc[mask_position, cols_position] = np.nan
    df.loc[mask_speed, cols_speed] = np.nan
    
    # remove samples until sensor is initialized and running properly
    # first_valid_idx = (~padded_mask).idxmax()
    # df = df.loc[first_valid_idx:].reset_index(drop=True)
    
    # interpolate Nan values
    if interpolate:
        df[cols_position + cols_speed] = df[cols_position + cols_speed].interpolate()

    # plt.figure(figsize=(12,6))
    # plt.plot(time, mask_speed)
    # plt.show()

    return df


# rotate sensor data by 180° around the specified axis
def rotate_sensor_data(df, rot_axis, acc_cols=('accX [g]', 'accY [g]', 'accZ [g]'), gyro_cols=('gyrX [dps]', 'gyrY [dps]', 'gyrZ [dps]')):

    rotated = df.copy()
    
    if rot_axis.lower() == 'x':
        rotated[acc_cols[1]] *= -1
        rotated[acc_cols[2]] *= -1
        rotated[gyro_cols[1]] *= -1
        rotated[gyro_cols[2]] *= -1
        
    elif rot_axis.lower() == 'y':
        rotated[acc_cols[0]] *= -1
        rotated[acc_cols[2]] *= -1
        rotated[gyro_cols[0]] *= -1
        rotated[gyro_cols[2]] *= -1
        
    elif rot_axis.lower() == 'z':
        rotated[acc_cols[0]] *= -1
        rotated[acc_cols[1]] *= -1
        rotated[gyro_cols[0]] *= -1
        rotated[gyro_cols[1]] *= -1

    else: 
        raise ValueError("Axis must be 'x', 'y' or 'z'.")

    return rotated


# compute the resultant acceleration of the imu sensor
def add_resultant_acc(df):
    df["accRes [g]"] = np.sqrt(df["accX [g]"]**2 + 
                            df["accY [g]"]**2 + 
                            df["accZ [g]"]**2)
    return df


# compute the resultant speed of the gnss sensor
def add_resultant_speed(df):
    df["speedRes [m/s]"] = np.sqrt(df["speedN [m/s]"]**2 +
                                   df["speedE [m/s]"]**2 +
                                   df["speedD [m/s]"]**2)
    return df


########################################################################
#         3. Session Detection   
########################################################################

# split measurements every time the timestamp is reset
def detect_sessions_imu(df, time_column = "imuTimestamp [us]"):

    # Ensure time column exists
    if time_column not in df.columns:
        raise ValueError(f"'{time_column}' column not found in dataset")

    # Detect resets in the time column
    resets = df[time_column].diff() < 0

    # Assign group numbers for each measurement, starting at 1
    session_ids = resets.cumsum() + 1

    # Split into groups
    df_sessions = {}
    sessions_summary = []
    
    for session_id, group in df.groupby(session_ids):

        # Extract start and end time
        start_us = group[time_column].iloc[0]
        end_us = group[time_column].iloc[-1]

        start_idx = group.index[0]
        end_idx = group.index[-1]               
        size = len(group)
        
        print(f"Session {session_id:02d} (IMU): last timestamp = {end_us} us, last row = {end_idx}, samples = {size}")
        
        # Append detected session
        session_key = f"Session{session_id:02d}"
        df_sessions[session_key] = group.reset_index(drop=True)

        # Create summary of session
        sessions_summary.append({
            "session_id": session_key,
            "samples_imu": size,
            "start_idx" : start_idx,
            "end_idx" : end_idx,
            "start_time_us" : start_us,
            "end_time_us": end_us            
        })

    return df_sessions, pd.DataFrame(sessions_summary)

            
# split measurements every time the timestamp is reset
def detect_sessions_gnss(df, time_column = "timestamp [us]"):

    # Ensure time column exists
    if time_column not in df.columns:
        raise ValueError(f"'{time_column}' column not found in dataset")

    # Detect resets in the time column
    resets = df[time_column].diff() < 0

    # Assign group numbers for each measurement, starting at 1
    session_ids = resets.cumsum() + 1

    # Split into groups
    df_sessions = {}
    sessions_summary = []
    
    for session_id, group in df.groupby(session_ids):

        # Extract start and end time
        start_us = group[time_column].iloc[0]
        end_us = group[time_column].iloc[-1]

        start_idx = group.index[0]
        end_idx = group.index[-1]               
        size = len(group)

        # Extract date and time of first sample
        run_start_posix = group['time [POSIXms]'].iloc[0]
        dt = datetime.fromtimestamp(run_start_posix / 1000)
        date_start = dt.date()
        time_start = dt.time()

        # Extract date and time of last sample
        run_end_posix = group['time [POSIXms]'].iloc[-1]
        dt = datetime.fromtimestamp(run_end_posix / 1000)
        date_end = dt.date()
        time_end = dt.time()
        
        print(f"Session {session_id:02d} (GNSS): last timestamp = {end_us} us, last row = {end_idx}, samples = {size}")
        
        # Append detected session
        session_key = f"Session{session_id:02d}"
        df_sessions[session_key] = group.reset_index(drop=True)

        # Create summary of session
        sessions_summary.append({
            "session_id": session_key,
            "start_date": date_start,
            "start_time_global": time_start,
            "end_date": date_end,
            "end_time_global": time_end,
            "samples_gnss": size,
            "start_idx" : start_idx,
            "end_idx" : end_idx,
            "start_time_us" : start_us,
            "end_time_us": end_us
        })

    return df_sessions, pd.DataFrame(sessions_summary)




########################################################################
#         4. Run Detection  
########################################################################

# detect number of runs per session
def detect_runs(df_gnss,
                v_start=5.0,            # threshold to start run
                v_hold=2.0,             # threshold to hold run
                alt_rise_end=20.0,      # altitude increase to end run
                alt_drop_min=50.0,      # min alt drop for run to be counted
                run_duration_min=20.0,  # min run duration in s for run to be counted
                len_window=10):         # window length for smoothing

    # Ensure numpy arrays
    posix_time = df_gnss['time [POSIXms]']
    time = df_gnss['timestamp [us]']
    alt = df_gnss['altitude [m]'].rolling(len_window, center=True, min_periods=1).mean()
    vel = df_gnss['speedRes [m/s]'].rolling(len_window, center=True, min_periods=1).mean()

    state = "IDLE"
    runs = []
    run_counter = 0

    for i in range(1, len(df_gnss)):
        dh = alt.iloc[i] - alt.iloc[i-1] 
        v = vel.iloc[i]
        if state == "IDLE":
            # look for start condition
            if v > v_start and dh < 0:
                state = "RUNNING"
                start_posix = posix_time.iloc[i]
                start_us = time.iloc[i]
                start_idx = df_gnss.index[df_gnss['timestamp [us]'] == start_us][0]
                start_alt = alt.iloc[i]
        elif state == "RUNNING":
            # check if skier stopped/fell
            if v < v_hold:
                state = "HOLD"
                hold_start_posix = posix_time.iloc[i]
                hold_start_us = time.iloc[i]
                hold_start_idx = df_gnss.index[df_gnss['timestamp [us]'] == hold_start_us][0]
                hold_start_alt = alt.iloc[i]
        elif state == "HOLD":
            # check if altitude has risen over threshold or if sensor was turned off --> run over
            if (alt[i] - hold_start_alt > alt_rise_end) or (i == len(df_gnss) - 1):
                state = "IDLE"
                end_posix = hold_start_posix
                end_us = hold_start_us
                end_idx = hold_start_idx
                duration = (end_us - start_us) / 1e6
                alt_drop = start_alt - hold_start_alt
                # add new valid run to list
                if ((alt_drop > alt_drop_min) and (duration > run_duration_min)):
                    run_counter += 1
                    run_key = f"Run{run_counter:02d}"
                    runs.append({
                            'run_id': run_key,
                            'start_time_global': datetime.fromtimestamp(start_posix / 1000).time(),
                            'end_time_global': datetime.fromtimestamp(end_posix / 1000).time(),
                            'duration_s': duration,
                            'alt_drop_m': alt_drop,
                            # 'start_idx': start_idx,
                            # 'end_idx': end_idx,
                            'start_time_us': start_us,
                            'end_time_us': end_us
                    })
            # if skier resumes moving, continue run
            elif v > v_start and dh < 0:
                state = "RUNNING"

    print(f"Detected runs: {len(runs)}")

    return pd.DataFrame(runs)


########################################################################
#         5. Jump Detection   
########################################################################

# compute time to peak
def get_landing_peak(df, land_idx, col='accY [g]', fs=200):
   
    offset = int(fs / 2)
    
    start = land_idx
    end = start + offset

    # find peak
    peak_idx = df.loc[start:end][col].abs().idxmax()
    peak = df.loc[peak_idx, col]

    return peak, peak_idx


# detect core airborne segments
def detect_core_segments(df_accVert, df_accRes, th_core_accVert, th_core_accRes, min_duration_s=0.05):

    # detect values below threshold, i.e. possible airborne samples
    core_air = (df_accVert < th_core_accVert) & (df_accRes < th_core_accRes)
    
    # detect transitions (true -> false and vice versa)
    transitions = core_air.astype(int).diff()
    core_starts = df_accVert.index[transitions == 1].tolist()
    core_ends = df_accVert.index[transitions == -1].tolist()

    # remove possible leading end that occurs before the first start
    if len(core_ends) > 0 and (len(core_starts) == 0 or core_ends[0] < core_starts[0]):
        core_ends.pop(0)

    # remove possible trailing start without matching end
    if len(core_starts) > len(core_ends):
        core_starts.pop() 

    # make list with pairs of start and end
    raw_segments = list(zip(core_starts, core_ends))

    # remove segments that are too short
    raw_segments = [(s, e) for (s, e) in raw_segments if (e - s) >= (min_duration_s * SAMPLE_RATE_IMU)]

    return raw_segments

    
# merge core airborne segments
def merge_core_segments(df_accVert, df_accRes, raw_segments, th_peak_accVert, th_peak_accRes, max_gap_s=0.3):
    merged_segments = [] 
    if len(raw_segments) == 0:
        return []
        
    current_start, current_end = raw_segments[0]
    
    for next_start, next_end in raw_segments[1:]:       
        gap_start = current_end
        gap_end = next_start
        gap_len = gap_end - gap_start
        df_vert = df_accVert[gap_start:gap_end]
        df_acc = df_accRes[gap_start:gap_end]

        # check if gap is short enough and if there is no peak in between
        if gap_len < (max_gap_s * SAMPLE_RATE_IMU) and df_vert.max() < th_peak_accVert and df_acc.max() < th_peak_accRes:
            current_end = next_end
        else:
            merged_segments.append((current_start, current_end))
            current_start = next_start
            current_end = next_end

    merged_segments.append((current_start, current_end))

    return merged_segments


# jump detection for laboratory measurements
def detect_jumps_lab(df, sensor_position, axis_vert, axis_long='accZ [g]', time_col='imuTimestamp [us]', th_core_accVert=0.5, th_core_accRes=0.8, th_crossing=1.0, window_size=10, min_duration_s=0.05, max_duration_s=5):

    # set threshold according to sensor position
    if sensor_position == 'Fuss':
        th_peak_vert = 5
        th_peak_long = 3
        th_peak_res = 6
    elif sensor_position == 'KSP':
        th_peak_vert = 4
        th_peak_long = 2.5
        th_peak_res = 4.5
    else:
        th_peak_vert = 2.5
        th_peak_long = 2
        th_peak_res = 3.5

    # copy df for processing
    df_smoothed = df.copy()

    # smooth the values
    df_smoothed[axis_vert] = df_smoothed[axis_vert].rolling(window_size, center=True).mean()
    df_smoothed['accRes [g]'] = df_smoothed['accRes [g]'].rolling(window_size, center=True).mean()

    # detect core segments
    raw_segments = detect_core_segments(df_smoothed[axis_vert], df_smoothed['accRes [g]'], th_core_accVert, th_core_accRes, min_duration_s=0.05)

    # merge core segments if: Time gap is short enough and no peak lies in between 
    merged_segments = merge_core_segments(df_smoothed[axis_vert], df_smoothed['accRes [g]'], raw_segments, th_peak_vert, th_peak_res, max_gap_s=0.1)
    
    # expand to full flight duration
    jumps = []
    used_landings = set()

    for start_idx, end_idx in merged_segments:
        # remove sequences that are too long
        if (end_idx - start_idx) > (SAMPLE_RATE_IMU * max_duration_s): 
            continue

        idx_t = df_smoothed.index.get_loc(start_idx)
        idx_l = df_smoothed.index.get_loc(end_idx)
        
        # add margins
        margin_acc = th_crossing / 5
        margin_time = int(2.5*SAMPLE_RATE_IMU)
        
        # make sure to find takeoff, i.e. where threshold is clearly crossed by a margin
        start = max(0, idx_t-margin_time)
        idx_confirm_takeoff = next((df_smoothed.index[j] for j in range(idx_t, start, -1) if df_smoothed.iloc[j]['accRes [g]'] >= th_crossing + margin_acc), None)

        # find actual takeoff
        if idx_confirm_takeoff is None:
            # no takeoff is detected
            continue
        else:
            # go forward and find first sample after takeoff (acc <= th_crossing)
            takeoff_idx = next((df_smoothed.index[j] for j in range(idx_confirm_takeoff, idx_l) if df_smoothed.iloc[j]['accRes [g]'] <= th_crossing), None)

        # make sure to find landing, i.e. where threshold is clearly crossed by a margin
        stop = min(len(df_smoothed), idx_l + margin_time)
        idx_confirm_landing = next((df_smoothed.index[j] for j in range(idx_l, stop) if df_smoothed.iloc[j]['accRes [g]'] >= th_crossing + 2* margin_acc), None)

        # find actual landing
        if idx_confirm_landing is None:
            # no landing is detected
            continue
        else:
            # go forward and find first sample after landing (acc >= th_crossing)
            landing_idx = next((df_smoothed.index[j] for j in range(idx_confirm_landing, -1, -1) if df_smoothed.iloc[j]['accRes [g]'] <= th_crossing), None)

        if landing_idx is None:
            continue

        # remove jumps that are too short
        if (landing_idx - takeoff_idx) < (SAMPLE_RATE_IMU * min_duration_s): 
            continue

        # detect peaks after landing
        peak_vert, peak_vert_idx = get_landing_peak(df, landing_idx, col=axis_vert)
        peak_long, peak_long_idx = get_landing_peak(df, landing_idx, col=axis_long)
        peak_res, peak_res_idx = get_landing_peak(df, landing_idx, col='accRes [g]')
        
        # no landing event if peaks are too small. Take resultant value for longitudinal axis.
        if (peak_vert < th_peak_vert) or (abs(peak_long) < th_peak_long) or (peak_res < th_peak_res):
            # print(landing_idx, f"landing time: {df.loc[landing_idx, time_col]*1e-6} s")
            # print(peak_vert, peak_vert_idx)
            # print(peak_long, peak_long_idx)
            # print(peak_res, peak_res_idx)
            continue
      
        # check if landing was already used
        if landing_idx in used_landings:
            continue
        if peak_res_idx in used_landings:
            continue

        # add landing
        used_landings.add(landing_idx)
        used_landings.add(peak_res_idx)

        # compute steepness from landing to peak -> delta_acc / delta_t
        delta_t = (peak_res_idx - landing_idx) / SAMPLE_RATE_IMU
        RFD_to_peak = (peak_vert - df[axis_vert].loc[landing_idx]) / delta_t

        # compute impulse from landing to peak, i.e. the area under the curve
        Impulse = np.trapezoid(df[axis_vert].loc[landing_idx:peak_res_idx+1], dx=1/SAMPLE_RATE_IMU)
        Impulse_net = np.trapezoid(df[axis_vert].loc[landing_idx:peak_res_idx+1] - 1.0, dx=1/SAMPLE_RATE_IMU) # net value -> subtract 1g
        
        # Compute flight time
        t_flight_s = (landing_idx - takeoff_idx) / SAMPLE_RATE_IMU
        
        jumps.append({
            'takeoff_idx': takeoff_idx,
            'landing_idx': landing_idx,
            'flight_time_s': t_flight_s,
            'time_to_peak': delta_t,
            'peak_vert_idx': peak_vert_idx,
            'peak_vert_g': peak_vert,
            'peak_long_idx': peak_long_idx,
            'peak_long_g': peak_long,
            'peak_res_idx': peak_res_idx,
            'peak_res_g': peak_res,
            'RFD_to_peak': RFD_to_peak,
            'Impulse_to_peak': Impulse,
            'Impulse_to_peak_net': Impulse_net,
        })

    return pd.DataFrame(jumps)


# jump detection tuned for on-snow measurements
def detect_jumps_snow(df, axis_vert, axis_long='accZ [g]', th_core_accVert=0.5, th_core_accRes=0.7, th_crossing=1.0, window_size=10, min_duration_s=0.05, max_duration_s=7):

    # set threshold for max peak height in between two core segments that can be overseen
    th_peak_vert = 2.0
    th_peak_res = 2.0

    # copy df for processing
    df_smoothed = df.copy()

    # 0) smooth the values
    df_smoothed[axis_vert] = df_smoothed[axis_vert].rolling(window_size, center=True).mean()
    df_smoothed['accRes [g]'] = df_smoothed['accRes [g]'].rolling(window_size, center=True).mean()

    # 1) detect core segments
    raw_segments = detect_core_segments(df_smoothed[axis_vert], df_smoothed['accRes [g]'], th_core_accVert, th_core_accRes, min_duration_s=0.2)

    # 2) merge core segments if: Time gap is short enough and no peak lies in between 
    merged_segments = merge_core_segments(df_smoothed[axis_vert], df_smoothed['accRes [g]'], raw_segments, th_peak_vert, th_peak_res, max_gap_s=0.5)
    
    # 3) expand to full flight duration
    jumps = []
    used_landings_idx = set()
    used_landings_peak = set()

    for start_idx, end_idx in merged_segments:
        # remove sequences that are too long
        if (end_idx - start_idx) > (SAMPLE_RATE_IMU * max_duration_s): 
            continue

        idx_t = df_smoothed.index.get_loc(start_idx)
        idx_l = df_smoothed.index.get_loc(end_idx)
        
        # add margins for acceleration and time
        margin_acc = th_crossing / 5
        margin_time = int(2.5*SAMPLE_RATE_IMU)
        
        # make sure to find takeoff, i.e. where threshold is clearly crossed by a margin
        idx_confirm_takeoff = next((df_smoothed.index[j] for j in range(idx_t, idx_t-margin_time, -1) if df_smoothed.iloc[j]['accRes [g]'] >= th_crossing + margin_acc), None)

        # find actual takeoff
        if idx_confirm_takeoff is None:
            # no takeoff is detected
            continue
        else:
            # go forward and find first sample after takeoff (acc <= th_crossing)
            takeoff_idx = next((df_smoothed.index[j] for j in range(idx_confirm_takeoff, idx_l) if df_smoothed.iloc[j]['accRes [g]'] <= th_crossing), None)

        # make sure to find landing, i.e. where threshold is clearly crossed by a margin
        idx_confirm_landing = next((df_smoothed.index[j] for j in range(idx_l, idx_l+margin_time) if df_smoothed.iloc[j]['accRes [g]'] >= th_crossing + 2 * margin_acc), None)

        # find actual landing
        if idx_confirm_landing is None:
            # no landing is detected
            continue
        else:
            # go forward and find first sample after landing (acc >= th_crossing)
            landing_idx = next((df_smoothed.index[j] for j in range(idx_confirm_landing, -1, -1) if df_smoothed.iloc[j]['accRes [g]'] <= th_crossing + margin_acc), None)

        # remove jumps that are too short
        if (landing_idx - takeoff_idx) < (SAMPLE_RATE_IMU * min_duration_s): 
            continue
                               
        # 4) detect peaks after landing
        peak_vert, peak_vert_idx = get_landing_peak(df, landing_idx, col=axis_vert)
        # peak_long, peak_long_idx = get_landing_peak(df, landing_idx, col=axis_long)
        peak_res, peak_res_idx = get_landing_peak(df, landing_idx, col='accRes [g]')

        # no landing event if peaks are too small
        if peak_res < 2.2:
            # print(landing_idx, f"landing time: {df.loc[landing_idx, time_col]*1e-6} s")
            # print(peak_vert, peak_vert_idx)
            # print(peak_long, peak_long_idx)
            # print(peak_res, peak_res_idx)
            continue
        
        # 5) check if landing was already used
        if (landing_idx in used_landings_idx) or (peak_res_idx in used_landings_peak):
            continue

        # add landing
        used_landings_idx.add(landing_idx)
        used_landings_peak.add(peak_res_idx)
        
        # create jump id
        jump_id = f"Jump{len(used_landings_peak):02d}"

        # 6) compute further landing parameters
        # steepness from landing to peak -> delta_acc / delta_t
        delta_t = (peak_res_idx - landing_idx) / SAMPLE_RATE_IMU
        RFD_to_peak = (peak_res - df['accRes [g]'].loc[landing_idx]) / delta_t

        # impulse from landing to peak, i.e. the area under the curve
        Impulse = np.trapezoid(df['accRes [g]'].loc[landing_idx:peak_res_idx+1], dx=1/SAMPLE_RATE_IMU)
        Impulse_net = np.trapezoid(df['accRes [g]'].loc[landing_idx:peak_res_idx+1] - 1.0, dx=1/SAMPLE_RATE_IMU) # net value -> subtract 1g
        
        # compute flight time
        t_flight_s = (landing_idx - takeoff_idx) / SAMPLE_RATE_IMU
        
        jumps.append({
            'jump_id': jump_id,
            'takeoff_idx': takeoff_idx,
            'landing_idx': landing_idx,
            'flight_time_s': t_flight_s,
            # 'peak_vert_idx': peak_vert_idx,
            # 'peak_vert_g': peak_vert,
            # 'peak_long_idx': peak_long_idx,
            # 'peak_long_g': peak_long,
            'peak_res_idx': peak_res_idx,
            'peak_res_g': peak_res,
            # 'time_to_peak_s': delta_t,
            # 'RFD_to_peak': RFD_to_peak,
            # 'Impulse_to_peak': Impulse,
            # 'Impulse_to_peak_net': Impulse_net,
        })

    return pd.DataFrame(jumps)
    
########################################################################
#         Plotting   
########################################################################

# plot imu data in single plots
def plot_imu(df, cols=None, folder="images/", series_name="imu_plot.png", show=True, saveFig=False):

    time = df[df.columns[0]] / 1e6

    labels = list(df.columns[1:])  # signal labels from column 1 onward

    # Default: plot all IMU columns except timestamp
    if cols is None:
        cols = list(range(1, len(df.columns)))

    n_plots = len(cols)
    if n_plots == 0:
        print("No valid columns selected. Nothing to plot.")
        return

    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots))
    if n_plots == 1:
        axs = [axs]

    for i, c in enumerate(cols):
        ax = axs[i]
        ax.plot(time, df[df.columns[c]], label=labels[c-1], linewidth=0.7)
        ax.set_ylabel(labels[c-1])
        ax.set_xlabel("Time [s]")
        # ax.legend(loc="upper right")
        ax.set_title(f"{series_name} - {labels[c-1]}")
        ax.grid(True)

    plt.tight_layout()

    if saveFig:
        image_path = folder / f"{series_name}_IMU.png"
        plt.savefig(image_path, dpi=300)
        print(f"Plot saved as {series_name}_IMU")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return

    
# plot imu data combined in two plots: acc + gyro
def plot_imu_combined(df, cols=None, folder="images/", series_name="imu_plot_combined.png", y_lim=None, show=True, saveFig=False):

    time = df[df.columns[0]] / 1e6
    
    labels = list(df.columns[1:])

    # Default: plot all IMU channels
    if cols is None:
        cols = list(range(1, 7))

    # Check if accel or gyro should be plotted
    plot_acc = any(c in [1, 2, 3] for c in cols)
    plot_gyro = any(c in [4, 5, 6] for c in cols)

    n_plots = plot_acc + plot_gyro
    if n_plots == 0:
        print("No valid columns selected. Nothing to plot.")
        return
    
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axs = [axs]  # make iterable if only one subplot

    plot_idx = 0

    # Accelerometer plot
    if plot_acc:
        ax = axs[plot_idx]
        for c in [1, 2, 3]:
            if c in cols:
                ax.plot(time, df[df.columns[c]], label=labels[c-1], linewidth=0.7)
        ax.set_ylabel("Acceleration [g]")
        ax.set_xlabel("Time [s]")
        if y_lim != None:
            ax.set_ylim([-y_lim,y_lim])
        ax.legend(loc="upper right")
        ax.set_title(f"{series_name} - Acceleration")
        plot_idx += 1

    # Gyroscope plot
    if plot_gyro:
        ax = axs[plot_idx]
        for c in [4, 5, 6]:
            if c in cols:
                ax.plot(time, df[df.columns[c]], label=labels[c-1], linewidth=0.7)
        ax.set_ylabel("Angular Velocity [dps]")
        ax.set_xlabel("Time [s]")
        ax.legend(loc="upper right")
        ax.set_title(f"{series_name} - Gyroscope")

    plt.tight_layout()

    if saveFig:
        image_path = folder / f"{series_name}_IMU_combined.png"
        plt.savefig(image_path, dpi=300)
        print(f"Plot saved as {series_name}_IMU_combined")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return



# plot gnss data
def plot_gnss(df, cols=None, folder="images/", series_name="gnss_plot.png", show=True, saveFig=False):

    # Use the second column as relative time in seconds
    time = df[df.columns[1]] / 1e6

    # signal labels from column 1 onward
    labels = list(df.columns[1:])  

    # Default: plot all GNSS columns except timestamp & time
    if cols is None:
        cols = list(range(2, len(df.columns)))

    n_plots = len(cols)
    if n_plots == 0:
        print("No valid columns selected. Nothing to plot.")
        return

    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axs = [axs]

    for i, c in enumerate(cols):
        ax = axs[i]
        ax.plot(time, df[df.columns[c]], label=labels[c-1], linewidth=0.7)
        ax.set_ylabel(labels[c-1])
        ax.set_xlabel("Time [s]")
        # ax.legend(loc="upper right")
        ax.set_title(f"{series_name} - {labels[c-1]}")
        ax.grid(True)

    plt.tight_layout()

    if saveFig:
        image_path = folder / f"{series_name}_GNSS.png"
        plt.savefig(image_path, dpi=300)
        print(f"Plot saved as {series_name}_GNSS")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return


def plot_runs(df_imu, df_gnss, df_runs, time_col='imuTimestamp [us]', folder="images/", series_name="runs", saveFig=False, show=True):

    time_imu = df_imu[time_col] / 1e6
    time_gnss = df_gnss["timestamp [us]"] / 1e6

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot acceleration signal
    ax1.plot(time_imu, df_imu['accRes [g]'], label='accRes [g]', linewidth=0.8)
    
    # if axis_vert != None:
    #     # ax1.plot(time_imu, df_imu[axis_vert], label=axis_vert, linewidth=0.8)
    #     # plt.plot(time_imu, df_imu['accZ [g]'], label='Acc Z', linewidth=0.8)
    #     ax1.plot(time_imu, df_imu['accRes [g]'], label='accRes [g]', linewidth=0.8)
    # else:
    #     ax1.plot(time_imu, df_imu['accX [g]'], label='accX [g]', linewidth=0.8)
    #     ax1.plot(time_imu, df_imu['accY [g]'], label='accY [g]', linewidth=0.8)
    #     ax1.plot(time_imu, df_imu['accZ [g]'], label='accZ [g]', linewidth=0.8)

    for _, row in df_runs.iterrows():
        # take GNSS timestamp and find nearest IMU timestamp
        start_ts = row['start_time_us'] / 1e6
        end_ts = row['end_time_us'] / 1e6
        start_idx = (time_imu - start_ts).abs().argmin()
        end_idx = (time_imu - end_ts).abs().argmin()
        
        # takeoff, landing and peak markers
        ax1.axvline(time_imu.iloc[start_idx], color='g', linestyle='--', linewidth=0.8, label='Start' if _ == df_runs.index[0] else "")
        ax1.axvline(time_imu.iloc[end_idx], color='r', linestyle='--', linewidth=0.8, label='End' if _ == df_runs.index[0] else "")
   
    # Labels and formatting
    ax1.set_xlim(0, time_imu.max())
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Acc [g]")
    ax1.set_title(f"{series_name} - Detected Runs")
    ax1.legend(loc='best')
    ax1.grid(True)

    # Plot altitude signal
    ax2.plot(time_gnss, df_gnss['altitude [m]'], linewidth=0.8)
    ax2.set_xlim(0, time_gnss.max())
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Altitude [m]")
    ax2.set_title(f"{series_name} - Altitude")
    ax2.grid(True)

    # Plot speed signal
    ax3.plot(time_gnss, df_gnss['speedRes [m/s]'], linewidth=0.8)
    ax3.set_xlim(0, time_gnss.max())
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Speed [m/s]")
    ax3.set_title(f"{series_name} - Resultant Speed")
    ax3.grid(True)

    plt.tight_layout()

    if saveFig:
        image_path = folder / f"{series_name}_DetectedRuns.png"
        plt.savefig(image_path, dpi=300)
        print(f"Plot saved as {series_name}_DetectedRuns")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return
    

# plot all detected jumps of a series
def plot_jumps(df_imu, df_gnss, df_jumps, axis_vert, th_crossing, time_col='imuTimestamp [us]', folder="images/", series_name="jumps", saveFig=False, show=True):

    time_imu = df_imu[time_col] / 1e6
    time_gnss = df_gnss["timestamp [us]"] / 1e6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot acceleration signal
    ax1.plot(time_imu, df_imu[axis_vert], label=axis_vert, linewidth=0.8)
    ax1.plot(time_imu, df_imu['accRes [g]'], label='accRes [g]', linewidth=0.8)

    
    # threshold line
    ax1.axhline(th_crossing, color='gray', linestyle='--', linewidth=0.8,  label='Threshold')

    for _, row in df_jumps.iterrows():
        # takeoff, landing and peak markers
        ax1.axvline(time_imu[row['takeoff_idx']], color='g', linestyle='--', linewidth=0.8, label='Takeoff' if _ == df_jumps.index[0] else "")
        ax1.axvline(time_imu[row['landing_idx']], color='r', linestyle='--', linewidth=0.8, label='Landing' if _ == df_jumps.index[0] else "")
        ax1.axvline(time_imu[row['peak_res_idx']], color='orange', linestyle='--', linewidth=0.8, label='Peak' if _ == df_jumps.index[0] else "")

    # set x-limits to capture interesting part of the jump
    # if not df_jumps.empty:
    #     margin = int(SAMPLE_RATE_IMU * 1.5)
    #     x_min = max(0, df_jumps['takeoff_idx'].min() - margin)
    #     x_max = min(len(df_imu) - 1, df_jumps['peak_res_idx'].max() + margin)
    #     time_min = time_imu[x_min]
    #     time_max = time_imu[x_max]
    #     plt.xlim(time_min, time_max)
        
    # Labels and formatting
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Acc [g]")
    ax1.set_title(f"{series_name} - Detected Jumps")
    ax1.legend(loc='best')
    ax1.grid(True)

    # Plot gnss signal
    ax2.plot(time_gnss, df_gnss['altitude [m]'], label='raw', linewidth=0.8)
    from scipy.signal import butter, filtfilt
    cutoff = 1
    fs = 10
    order = 4
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, df_gnss['altitude [m]'])
    
    window_size = 10
    ax2.plot(time_gnss, y, label=f'LP filter', linewidth=0.8)
    # ax2.plot(time_gnss, df_gnss['altitude [m]'].rolling(10, center=True, min_periods=1).mean(), label=f'movAvg (n={window_size})', linewidth=0.8)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Altitude [m]")
    ax2.legend(loc='best')
    ax2.set_title(f"{series_name} - Altitude")
    ax2.grid(True)

    plt.tight_layout()

    if saveFig:
        image_path = folder / f"{series_name}_Jumps.png"
        plt.savefig(image_path, dpi=300)
        print(f"Plot saved as {series_name}_Jumps")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return


# plot each jump of a series in a separate plot
def plot_jumps_separately(df_imu, df_gnss, df_jumps, axis_vert, th_crossing, time_col='imuTimestamp [us]', folder="images/", series_name="jumps", saveFig=False, show=True):

    time_imu = df_imu[time_col] / 1e6
    time_gnss = df_gnss['timestamp [us]'] / 1e6
    
    # takeoff & landing markers
    for i, (_, row) in enumerate(df_jumps.iterrows(), start=1):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # set time window to capture interesting part of the jump
        margin = int(SAMPLE_RATE_IMU * 0.75)
        idx_min = max(0, row['takeoff_idx'] - margin)
        idx_max = min(len(df_imu) - 1, row['peak_res_idx'] + margin)
        t_min = time_imu.loc[idx_min]
        t_max = time_imu.loc[idx_max]

        # time masks
        imu_mask = (time_imu >= t_min) & (time_imu <= t_max)
        gnss_mask = (time_gnss >= t_min) & (time_gnss <= t_max)

         # plot acceleration signal
        if axis_vert != None:
            # ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, axis_vert], label=axis_vert, linewidth=0.8)
            # plt.plot(time_imu, df['accZ [g]'], label='Acc Z', linewidth=0.8)
            ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, 'accRes [g]'], label='accRes [g]', linewidth=0.8)
        else:
            ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, 'accX [g]'], label='accX [g]', linewidth=0.8)
            ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, 'accY [g]'], label='accY [g]', linewidth=0.8)
            ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, 'accZ [g]'], label='accZ [g]', linewidth=0.8)
        
        # threshold line
        ax1.axhline(th_crossing, color='gray', linestyle='--', linewidth=0.8,  label='Threshold')
        
        # takeoff, landing and peak markers
        ax1.axvline(time_imu[row['takeoff_idx']], color='g', linestyle='--', linewidth=0.8, label='Takeoff')
        ax1.axvline(time_imu[row['landing_idx']], color='r', linestyle='--', linewidth=0.8, label='Landing')
        ax1.axvline(time_imu[row['peak_res_idx']], color='black', linestyle='--', linewidth=0.8, label='Peak')
        
         # fill area under curve from landing → peak
        # idx_range = np.arange(row['landing_idx'], row['peak_res_idx'] + 1)
        # plt.fill_between(time_imu[idx_range], df_imu['accRes [g]'].loc[idx_range], color='bisque', alpha=0.3, label='Impulse to peak')

        # Labels and formatting
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Acc [g]")
        ax1.set_title(f"{series_name} - Jump{i:02d}")
        ax1.legend(loc='best')
        ax1.grid(True)

        # Plot gnss signal
        ax2.plot(time_gnss[gnss_mask], df_gnss.loc[gnss_mask, "altitude [m]"], label='raw', linewidth=0.8)
        window_size = 10
        ax2.plot(time_gnss[gnss_mask], df_gnss.loc[gnss_mask, "altitude [m]"].rolling(window_size, center=True, min_periods=1).mean(), label=f'movAvg (n={window_size})', linewidth=0.8)
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Altitude [m]")
        ax2.set_title(f"{series_name} - Jump{i:02d}")
        ax2.legend(loc='best')
        ax2.grid(True)
        
        plt.tight_layout()
    
        if saveFig:
            plot_name = f"{series_name}_Jump{i:02d}"
            image_path = folder / f"{plot_name}.png"
            plt.savefig(image_path, dpi=300)
            print(f"Plot saved as {plot_name}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)

    return

# plot all detected jumps of a series without gnss data
def plot_jumps_lab(df_imu, df_jumps, axis_vert, th_crossing, time_col='imuTimestamp [us]', folder="images/", series_name="jumps", saveFig=False, show=True):

    time_imu = df_imu[time_col] / 1e6

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))

    # Plot acceleration signal
    # ax1.plot(time_imu, df_imu[axis_vert], label=axis_vert, linewidth=0.8)
    ax1.plot(time_imu, df_imu['accRes [g]'], label='accRes [g]', linewidth=0.8)
    
    # threshold line
    ax1.axhline(th_crossing, color='gray', linestyle='--', linewidth=0.8,  label='Threshold')

    for _, row in df_jumps.iterrows():
        # takeoff, landing and peak markers
        ax1.axvline(time_imu[row['takeoff_idx']], color='g', linestyle='--', linewidth=0.8, label='Takeoff' if _ == df_jumps.index[0] else "")
        ax1.axvline(time_imu[row['landing_idx']], color='r', linestyle='--', linewidth=0.8, label='Landing' if _ == df_jumps.index[0] else "")
        ax1.axvline(time_imu[row['peak_res_idx']], color='orange', linestyle='--', linewidth=0.8, label='Peak' if _ == df_jumps.index[0] else "")

    # set x-limits to capture interesting part of the jump
    # if not df_jumps.empty:
    #     margin = int(SAMPLE_RATE_IMU * 1.5)
    #     x_min = max(0, df_jumps['takeoff_idx'].min() - margin)
    #     x_max = min(len(df_imu) - 1, df_jumps['peak_res_idx'].max() + margin)
    #     time_min = time_imu[x_min]
    #     time_max = time_imu[x_max]
    #     plt.xlim(time_min, time_max)
        
    # Labels and formatting
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Acc [g]")
    ax1.set_title(f"{series_name} - Detected Jumps")
    ax1.legend(loc='best')
    ax1.grid(True)

    plt.tight_layout()

    if saveFig:
        image_path = folder / f"{series_name}_Jumps.png"
        plt.savefig(image_path, dpi=300)
        print(f"Plot saved as {series_name}_Jumps")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return



# plot each jump of a series in a separate plot without gnss data
def plot_jumps_separately_lab(df_imu, df_jumps, axis_vert, th_crossing, time_col='imuTimestamp [us]', folder="images/", series_name="jumps", saveFig=False, show=True):

    time_imu = df_imu[time_col] / 1e6
    
    # takeoff & landing markers
    for i, (_, row) in enumerate(df_jumps.iterrows(), start=1):
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))

        # set time window to capture interesting part of the jump
        margin = int(SAMPLE_RATE_IMU * 0.75)
        idx_min = max(0, row['takeoff_idx'] - margin)
        idx_max = min(len(df_imu) - 1, row['peak_res_idx'] + margin)
        t_min = time_imu.loc[idx_min]
        t_max = time_imu.loc[idx_max]

        # time masks
        imu_mask = (time_imu >= t_min) & (time_imu <= t_max)

         # plot acceleration signal
        if axis_vert != None:
            # ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, axis_vert], label=axis_vert, linewidth=0.8)
            # plt.plot(time_imu, df['accZ [g]'], label='Acc Z', linewidth=0.8)
            ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, 'accRes [g]'], label='accRes [g]', linewidth=0.8)
        else:
            ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, 'accX [g]'], label='accX [g]', linewidth=0.8)
            ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, 'accY [g]'], label='accY [g]', linewidth=0.8)
            ax1.plot(time_imu[imu_mask], df_imu.loc[imu_mask, 'accZ [g]'], label='accZ [g]', linewidth=0.8)
        
        # threshold line
        ax1.axhline(th_crossing, color='gray', linestyle='--', linewidth=0.8,  label='Threshold')
        
        # takeoff, landing and peak markers
        ax1.axvline(time_imu[row['takeoff_idx']], color='g', linestyle='--', linewidth=0.8, label='Takeoff')
        ax1.axvline(time_imu[row['landing_idx']], color='r', linestyle='--', linewidth=0.8, label='Landing')
        ax1.axvline(time_imu[row['peak_res_idx']], color='black', linestyle='--', linewidth=0.8, label='Peak')
        
         # fill area under curve from landing → peak
        # idx_range = np.arange(row['landing_idx'], row['peak_res_idx'] + 1)
        # plt.fill_between(time_imu[idx_range], df_imu['accRes [g]'].loc[idx_range], color='bisque', alpha=0.3, label='Impulse to peak')

        # Labels and formatting
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Acc [g]")
        ax1.set_title(f"{series_name} - Jump{i:02d}")
        ax1.legend(loc='best')
        ax1.grid(True)

        plt.tight_layout()
    
        if saveFig:
            plot_name = f"{series_name}_Jump{i:02d}"
            image_path = folder / f"{plot_name}.png"
            plt.savefig(image_path, dpi=300)
            print(f"Plot saved as {plot_name}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)

    return