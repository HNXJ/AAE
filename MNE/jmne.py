import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import h5py
import mne
import re # regex
import os

from scipy import stats
from statsmodels.stats import multitest
from scipy.ndimage import label, generate_binary_structure


def _extract_string_list_from_mat(matlab_array_or_list):
    """
    Helper function to extract a list of strings from a MATLAB cell array or numpy object array.
    This function is primarily for scipy.io.loadmat output or lists of strings.
    """
    string_list = []
    if isinstance(matlab_array_or_list, list):
        for item in matlab_array_or_list:
            if isinstance(item, (str, np.str_)):
                string_list.append(str(item))
            elif isinstance(item, (bytes, np.bytes_)): # Decode byte strings
                string_list.append(item.decode('utf-8'))
            elif isinstance(item, list): # Handle nested lists if present
                string_list.extend(_extract_string_list_from_mat(item))
            elif isinstance(item, np.ndarray) and item.size == 1 and item.dtype.kind in ('U', 'S', 'O'):
                string_list.append(str(item.item()))
            elif isinstance(item, np.ndarray) and item.dtype == np.uint16: # Raw MATLAB char array
                 string_list.append(''.join(chr(c) for c in item.flatten() if c > 0).strip())
            else:
                string_list.append(str(item))
    elif isinstance(matlab_array_or_list, np.ndarray) and matlab_array_or_list.dtype.kind == 'O':
        for item_wrapper in matlab_array_or_list.flatten():
            if isinstance(item_wrapper, np.ndarray) and item_wrapper.size > 0:
                for sub_item in item_wrapper.flatten():
                    if isinstance(sub_item, (str, np.str_)):
                        string_list.append(str(sub_item))
                    elif isinstance(sub_item, np.ndarray) and sub_item.size == 1 and sub_item.dtype.kind in ('U', 'S', 'O'):
                        string_list.append(str(sub_item.item()))
            elif isinstance(item_wrapper, (str, np.str_)):
                string_list.append(str(item_wrapper))
            elif isinstance(item_wrapper, (bytes, np.bytes_)):
                string_list.append(item_wrapper.decode('utf-8'))
    elif isinstance(matlab_array_or_list, (str, np.str_)):
        string_list.append(str(matlab_array_or_list))
    elif isinstance(matlab_array_or_list, (bytes, np.bytes_)):
        string_list.append(matlab_array_or_list.decode('utf-8'))

    return string_list


def _load_mat_with_h5py(file_path):
    """
    Attempts to load a .mat file using h5py, suitable for MATLAB v7.3 files.
    Returns a dictionary of data with MATLAB structs/cell arrays converted to Python dicts/lists.
    """
    data = {}
    with h5py.File(file_path, 'r') as f:
        def _parse_h5py_object_internal(h5_obj):
            if isinstance(h5_obj, h5py.Group):
                parsed_dict = {}
                for key, value_obj in h5_obj.items():
                    parsed_dict[key] = _parse_h5py_object_internal(value_obj)
                return parsed_dict
            elif isinstance(h5_obj, h5py.Dataset):
                value = h5_obj[()]
                matlab_class = h5_obj.attrs.get('MATLAB_class')

                if h5_obj.dtype == h5py.ref_dtype:
                    # Array of references. Resolve each one.
                    return [_parse_h5py_object_internal(f[ref_val]) for ref_val in value.flat]
                elif matlab_class == 'cell':
                    if isinstance(value, np.ndarray) and value.dtype == h5py.ref_dtype:
                        return [_parse_h5py_object_internal(f[ref_val]) for ref_val in value.flat]
                    else: # Array of direct values, potentially mixed types
                        return [_parse_h5py_object_internal(item) for item in value.flat]
                elif matlab_class == 'char':
                    return ''.join(chr(c) for c in value.flatten() if c > 0).strip()
                elif isinstance(value, np.ndarray):
                    # Explicitly handle uint16 arrays as MATLAB char arrays
                    if value.dtype == np.uint16 and value.ndim <= 2:
                        return ''.join(chr(c) for c in value.flatten() if c > 0).strip()
                    elif value.dtype.kind == 'S':
                        return value.item().decode('utf-8') if value.size == 1 else [s.decode('utf-8') for s in value.flatten()]
                    elif value.dtype.kind == 'U':
                        return value.item() if value.size == 1 else value.flatten().tolist()
                    elif value.ndim == 0:
                        return value.item()
                    else:
                        return value
                elif isinstance(value, (bytes, str)):
                    return value.decode('utf-8') if isinstance(value, bytes) else value
                else:
                    return value # Scalar native Python types
            elif isinstance(h5_obj, h5py.Reference):
                return _parse_h5py_object_internal(f[h5_obj])
            return h5_obj

        for key in f.keys():
            if key == '#refs#':
                continue
            data[key] = _parse_h5py_object_internal(f[key])
    return data


def load_subject_metadata(subject_folder_path):
    """
    Loads subject-specific metadata (e.g., chan_names) from 'ECoGInfo.mat'
    within the specified subject data folder, handling MATLAB v7.3 files.

    Args:
        subject_folder_path (str): The path to the subject's data folder (e.g., UM1403_Oddball_Bip).

    Returns:
        dict: A dictionary containing subject-specific metadata, or None if loading fails.
    """
    ecog_info_path = os.path.join(subject_folder_path, 'ECoGInfo.mat')
    loaded_info = None

    try:
        loaded_info = _load_mat_with_h5py(ecog_info_path)
        print(f"Loaded ECoGInfo.mat with h5py from {ecog_info_path}")
        output_struct = loaded_info.get('OUTPUT', loaded_info) # Fallback to top-level if 'OUTPUT' isn't key

    except Exception as h5_e:
        print(f"h5py failed to load {ecog_info_path}, attempting scipy.io.loadmat. Error: {h5_e}")
        try:
            loaded_info = sio.loadmat(ecog_info_path)
            print(f"Loaded ECoGInfo.mat with scipy.io.loadmat from {ecog_info_path}")
            output_struct_raw = loaded_info.get('OUTPUT', None)

            output_struct = output_struct_raw
            if isinstance(output_struct_raw, np.ndarray) and output_struct_raw.size == 1 and output_struct_raw.dtype.kind == 'V':
                output_struct = output_struct_raw.item()
            elif output_struct_raw is None:
                output_struct = loaded_info

        except FileNotFoundError:
            print(f"Error: ECoGInfo.mat not found at {ecog_info_path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading ECoGInfo.mat with scipy.io.loadmat: {e}")
            return None

    if loaded_info is None or output_struct is None:
        return None

    subject_metadata = {}

    if isinstance(output_struct, dict):
        temp_output_metadata = output_struct
    elif hasattr(output_struct, 'dtype') and output_struct.dtype.names:
        temp_output_metadata = {name: output_struct[name] for name in output_struct.dtype.names}
    else:
        print(f"Warning: 'OUTPUT' structure not recognized after loading {ecog_info_path}. Cannot extract metadata.")
        return None

    subject_metadata['chan_names'] = _extract_string_list_from_mat(temp_output_metadata.get('chan_names', []))
    subject_metadata['srate'] = temp_output_metadata.get('srate')
    subject_metadata['Conditions'] = _extract_string_list_from_mat(temp_output_metadata.get('Conditions', []))

    cond_nums_raw = temp_output_metadata.get('CondNums')
    if isinstance(cond_nums_raw, np.ndarray):
        subject_metadata['CondNums'] = cond_nums_raw.flatten().tolist()
    elif isinstance(cond_nums_raw, list): # From h5py parsing
         subject_metadata['CondNums'] = cond_nums_raw
    else:
        subject_metadata['CondNums'] = cond_nums_raw

    # --- MODIFICATION FOR STIM AND STIM_TIMING --- 
    stim_raw = temp_output_metadata.get('stim', None)
    
    # Initialize to None
    subject_metadata['stim'] = None
    subject_metadata['stim_timing'] = None

    if isinstance(stim_raw, np.ndarray):
        # Case 1: (N_events, 2) - standard MATLAB output for [timing, condition]
        if stim_raw.ndim == 2 and stim_raw.shape[1] == 2:
            subject_metadata['stim_timing'] = stim_raw[:, 0].flatten().astype(int) # Timing values
            subject_metadata['stim'] = stim_raw[:, 1].flatten().astype(int)         # Condition values
            print(f"Debug: Extracted stim_timing and stim from (N, 2) array (original shape: {stim_raw.shape}).")
        # Case 2: (2, N_events) - where row 0 is timing, row 1 is condition (observed in kernel state)
        elif stim_raw.ndim == 2 and stim_raw.shape[0] == 2:
            subject_metadata['stim_timing'] = stim_raw[0, :].flatten().astype(int) # Timing values
            subject_metadata['stim'] = stim_raw[1, :].flatten().astype(int)         # Condition values
            print(f"Debug: Extracted stim_timing and stim from (2, N) array (original shape: {stim_raw.shape}).")
        # Case 3: (1, 2*N_events) or (2*N_events,) - interleaved timing and condition in a flattened array
        elif (stim_raw.ndim == 2 and stim_raw.shape[0] == 1 and stim_raw.shape[1] % 2 == 0) or \
             (stim_raw.ndim == 1 and stim_raw.size % 2 == 0):
            processed_stim = stim_raw.flatten()
            reshaped_stim = processed_stim.reshape(-1, 2)
            subject_metadata['stim_timing'] = reshaped_stim[:, 0].flatten().astype(int)
            subject_metadata['stim'] = reshaped_stim[:, 1].flatten().astype(int)
            print(f"Debug: Extracted stim_timing and stim from interleaved array (original shape: {stim_raw.shape}).")
        else:
            print(f"Warning: stim_raw np.ndarray has unexpected shape: {stim_raw.shape}. Setting stim and stim_timing to None.")
    elif isinstance(stim_raw, list):
        # Handle list of [timing, condition] pairs
        if stim_raw and isinstance(stim_raw[0], (list, tuple, np.ndarray)) and len(stim_raw[0]) == 2:
            timing_list = [item[0] for item in stim_raw]
            cond_list = [item[1] for item in stim_raw]
            subject_metadata['stim_timing'] = np.array(timing_list).flatten().astype(int)
            subject_metadata['stim'] = np.array(cond_list).flatten().astype(int)
            print("Debug: Successfully extracted stim_timing and stim from list of pairs.")
        else:
            print(f"Warning: Unexpected stim_raw list format. Setting stim and stim_timing to None. (type: {type(stim_raw)})")
    else:
        print("Warning: 'stim' data not found or is in an unhandled format. Setting stim and stim_timing to None.")

    if not subject_metadata.get('chan_names'):
        print("Warning: 'chan_names' not found or empty in ECoGInfo.mat. This might cause issues with channel data loading.")

    print(f"Subject-specific metadata from ECoGInfo.mat: {list(subject_metadata.keys())}")
    print(f"  - Chan names (first 5): {subject_metadata.get('chan_names', [])[:5]}")

    return subject_metadata


def load_subject_channel_data(subject_folder_path, subject_chan_names):
    """
    Loads channel-specific continuous data for a subject from individual .mat files.

    Args:
        subject_folder_path (str): The path to the subject's data folder (e.g., base_path/UM1403_Oddball_Bip).
        subject_chan_names (list): A list of channel names (strings) specific to this subject's data.

    Returns:
        dict: A dictionary where keys are channel names and values are their 'continuous' data arrays.
              Includes error messages for channels that failed to load.
    """
    subject_raw_data = {}
    print(f"Attempting to load channel data from: {subject_folder_path}")

    for channel_name in subject_chan_names:
        channel_file_path = os.path.join(subject_folder_path, f'{channel_name}.mat')

        channel_raw_data = None
        try:
            channel_raw_data = _load_mat_with_h5py(channel_file_path)
        except Exception as h5_e:
            # print(f"h5py failed to load {channel_file_path}, attempting scipy.io.loadmat. Error: {h5_e}") # Commented to reduce verbosity
            try:
                channel_raw_data = sio.loadmat(channel_file_path)
            except FileNotFoundError:
                print(f"Error: Channel file {channel_file_path} not found. Skipping channel {channel_name}.")
                subject_raw_data[channel_name] = {'error': 'FileNotFound'}
                continue
            except Exception as e:
                print(f"An error occurred while loading channel {channel_name} data with scipy.io.loadmat: {e}. Skipping channel.")
                subject_raw_data[channel_name] = {'error': str(e)}
                continue

        if channel_raw_data is None:
            print(f"Error: Failed to load channel {channel_name} data from {channel_file_path}. Skipping channel.")
            subject_raw_data[channel_name] = {'error': 'LoadFailed'}
            continue

        data_struct = channel_raw_data.get('DATA', channel_raw_data)

        continuous_data = None
        if isinstance(data_struct, dict):
            continuous_data = data_struct.get('continuous', None)
        elif hasattr(data_struct, 'dtype') and data_struct.dtype.names:
            if 'continuous' in data_struct.dtype.names:
                continuous_data = data_struct['continuous']

        if continuous_data is not None:
            subject_raw_data[channel_name] = continuous_data
        else:
            print(f"Warning: 'continuous' data not found or could not be extracted for channel {channel_name} from {channel_file_path}. Skipping.")
            subject_raw_data[channel_name] = {'error': "'continuous' data not found or extractable"}

    return subject_raw_data


def create_mne_epochs(subject_raw_data, subject_metadata, tmin=-1.0, tmax=1.0):
    """
    Creates an MNE Info object, an MNE RawArray, identifies events, and generates
    an MNE Epochs object for a single subject.

    Args:
        subject_raw_data (dict): Dictionary of continuous data for each channel.
                                 Keys are channel names, values are numpy arrays.
        subject_metadata (dict): Dictionary containing subject-specific metadata
                                 like srate, chan_names, Conditions, CondNums, stim, and stim_timing.
        tmin (float): Start time of epochs relative to event in seconds.
        tmax (float): End time of epochs relative to event in seconds.

    Returns:
        mne.Epochs: An MNE Epochs object containing the epoched data.
    """

    # 3a. Filter subject_raw_data to include only successfully loaded channels
    #     and extract channel names.
    valid_channels_data = {}
    ch_names = []
    for chan_name, data in subject_raw_data.items():
        # Check if data is a numpy array and not an error entry (which would be a dict)
        if isinstance(data, np.ndarray):
            valid_channels_data[chan_name] = data
            ch_names.append(chan_name)

    if not ch_names:
        raise ValueError("No valid channels found in subject_raw_data.")

    # 3b. Get the sampling rate (sfreq) from subject_metadata['srate']
    sfreq = subject_metadata['srate']
    # Ensure sfreq is a scalar, not an array if it came from MATLAB
    if isinstance(sfreq, np.ndarray) and sfreq.size == 1:
        sfreq = sfreq.item()
    if not isinstance(sfreq, (int, float)):
        raise ValueError(f"Sampling rate (srate) is not a scalar: {sfreq}")

    # 3c. Create a list of channel types (ch_types) for all ch_names
    ch_types = ['ecog'] * len(ch_names)

    # 3d. Create an MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # 3e. Concatenate the continuous data from all valid channels
    # Ensure data is (n_channels, n_times) and convert to Volts
    concatenated_data_list = []
    for chan_name in ch_names:
        chan_data = valid_channels_data[chan_name]
        # Ensure data is 2D, e.g., (n_samples, 1) or (n_samples,)n_samples,) to (1, n_samples)
        if chan_data.ndim == 1:
            chan_data = chan_data[np.newaxis, :]
        elif chan_data.ndim == 2 and chan_data.shape[1] == 1: # (n_samples, 1) to (1, n_samples)
            chan_data = chan_data.T
        elif chan_data.ndim == 2 and chan_data.shape[0] != 1 and chan_data.shape[1] != 1: # Assume (n_times, n_features) or (n_features, n_times)
            # MNE expects (n_channels, n_times). If it's (n_times, n_channels) we need to transpose.
            # For now, assuming single channel data is (n_times, 1) or (n_times,)
            raise ValueError(f"Unexpected channel data shape for {chan_name}: {chan_data.shape}")
        elif chan_data.ndim > 2:
            raise ValueError(f"Unexpected channel data shape for {chan_name}: {chan_data.shape}")

        # MNE expects data in Volts. Assuming input is microVolts, convert to Volts.
        # This is a common assumption in ECoG. If not, adjustment might be needed.
        concatenated_data_list.append(chan_data * 1e-6)

    if not concatenated_data_list:
        raise ValueError("No data to concatenate after filtering.")

    # Stack all channel data. Resulting shape should be (n_channels, n_times)
    concatenated_data = np.vstack(concatenated_data_list)

    # 3f. Create an MNE RawArray object
    raw = mne.io.RawArray(data=concatenated_data, info=info, verbose=False)

    # 3g. Extract stim_timing and stim (condition codes)
    stim_timing_raw = subject_metadata['stim_timing']
    stim_raw = subject_metadata['stim']

    if isinstance(stim_timing_raw, np.ndarray) and stim_timing_raw.ndim == 1:
        stim_timing = stim_timing_raw
    elif isinstance(stim_timing_raw, np.ndarray) and stim_timing_raw.ndim == 2 and stim_timing_raw.shape[0] == 1:
        stim_timing = stim_timing_raw.flatten()
    else:
        raise ValueError(f"Unexpected format for stim_timing: {type(stim_timing_raw)} with shape {stim_timing_raw.shape if isinstance(stim_timing_raw, np.ndarray) else 'N/A'}")

    if isinstance(stim_raw, np.ndarray) and stim_raw.ndim == 1:
        stim = stim_raw
    elif isinstance(stim_raw, np.ndarray) and stim_raw.ndim == 2 and stim_raw.shape[0] == 1:
        stim = stim_raw.flatten()
    else:
        raise ValueError(f"Unexpected format for stim: {type(stim_raw)} with shape {stim_raw.shape if isinstance(stim_raw, np.ndarray) else 'N/A'}")

    # 3h. Create an MNE-compatible events array
    # Events array: (sample_index, 0, event_id)
    events = np.zeros((len(stim_timing), 3), dtype=int)
    events[:, 0] = stim_timing.astype(int) # Sample index
    events[:, 2] = stim.astype(int)        # Event ID

    # 3i. Create an event_id dictionary
    conditions = subject_metadata['Conditions']
    cond_nums = subject_metadata['CondNums']

    # Ensure cond_nums are integers
    cond_nums_int = [int(num) for num in cond_nums]

    event_id = dict(zip(conditions, cond_nums_int))

    # 3j. Generate an MNE Epochs object
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False,
                        event_repeated='merge') # Added parameter to handle non-unique events

    # 3k. Return the created epochs object
    return epochs


def compute_ersp(specs, data, output):
    """
    Replicates the ERSP computation logic found in `ComputeERSP.m` MATLAB function.

    Args:
        specs (dict): Dictionary representing MATLAB's SPECS structure.
                      Expected keys: 'frex', 'cyclex'.
        data (dict): Dictionary representing MATLAB's DATA structure.
                     Expected keys: 'continuous' (numpy array).
        output (dict): Dictionary representing MATLAB's OUTPUT structure.
                       Expected keys: 'srate' (float or int).

    Returns:
        dict: A dictionary containing 'ersp' (computed ERSP data) and 'raw' (raw complex spectral data).
    """
    num_frex = len(specs['frex'])
    srate = output['srate']
    continuous_data = data['continuous']

    # Ensure continuous_data is a 1D numpy array for processing
    if isinstance(continuous_data, np.ndarray):
        if continuous_data.ndim == 2 and continuous_data.shape[1] == 1:
            continuous_data = continuous_data.flatten()
        elif continuous_data.ndim > 1 and continuous_data.shape[0] == 1:
            continuous_data = continuous_data.flatten()
        elif continuous_data.ndim > 1:
            raise ValueError("continuous_data must be 1D or (N,1) or (1,N) array.")
    else:
        raise TypeError("continuous_data must be a numpy array.")

    # MATLAB's continuous data might be in different units (e.g., microvolts).
    # The ERSP calculation itself is unitless for normalization, but ensure consistency.

    # Create full time representation (MATLAB: 1/OUTPUT.srate:1/OUTPUT.srate:length(DATA.continuous)/OUTPUT.srate)
    time = np.arange(1/srate, len(continuous_data)/srate + 1/srate/2, 1/srate)

    # Wavelet analyses require time to be symmetrical
    wave_time = time - np.mean(time)

    # Setup fft
    EEGpnts = len(continuous_data)
    EEGtrials = 1  # Assuming continuous_data is for a single continuous recording

    # Define convolution parameters
    n_wavelet = len(time)
    n_data = EEGpnts * EEGtrials
    n_convolution = n_wavelet + n_data - 1
    n_conv_pow2 = 2**np.ceil(np.log2(n_convolution)).astype(int)
    half_of_wavelet_size = (n_wavelet - 1) // 2

    # Get FFT of data (reshape(continuous_data,1,EEGpnts*EEGtrials) is like flattening)
    eegfft = np.fft.fft(continuous_data, n_conv_pow2)

    # Initialize output arrays
    ersp_data = np.zeros((num_frex, len(continuous_data)), dtype=np.float64)
    raw_data = np.zeros((num_frex, len(continuous_data)), dtype=np.complex128)

    # Loop through frequencies and filter
    s = specs['cyclex'] / (2 * np.pi * specs['frex'])

    for fi in range(num_frex):
        # Define Morlet wavelet
        # fwave = exp(2*1i*pi*SPECS.frex(fi).*wave_time) .* exp(-wave_time.^2./(2*(s(fi)^2))) ;
        fwave = np.exp(2j * np.pi * specs['frex'][fi] * wave_time) * np.exp(-wave_time**2 / (2 * s[fi]**2))

        # FFT of wavelet
        wavelet_fft = np.fft.fft(fwave, n_conv_pow2)

        # Convolution in frequency domain (multiplication) and inverse FFT
        eegconv = np.fft.ifft(wavelet_fft * eegfft)

        # Remove padding (MATLAB: eegconv = eegconv(half_of_wavelet_size+1:end-half_of_wavelet_size);)
        # Python indices are 0-based, so +1 becomes just 'half_of_wavelet_size'
        # and 'end - half_of_wavelet_size' becomes 'eegconv.size - half_of_wavelet_size'
        eegconv_trimmed = eegconv[half_of_wavelet_size : eegconv.size - half_of_wavelet_size]

        # Reshape to original data length for consistency with single trial
        sig1 = eegconv_trimmed[:len(continuous_data)] # Take only relevant part if e.g. floating point error

        # Store values
        # DATA.ersp(fi,:) = sig1.*conj(sig1); # same as mean(abs(sig1).^2,2);
        ersp_data[fi, :] = np.abs(sig1)**2
        # DATA.raw(fi,:) = sig1;
        raw_data[fi, :] = sig1

    return {
        'ersp': ersp_data,
        'raw': raw_data
    }


def plot_line_figure(
    zerotime, num_conds, avg_data, chan_idx, xaxis_indices, plot_conds,
    color_array, line_thickness, figure_name, time_vector, yaxis_text, specs, legend_array
):
    """
    Replicates the plotting logic found in `ECOG_plotter_dep_figures_line_v3.m` MATLAB function.

    Args:
        zerotime (int): Index in time_vector corresponding to time zero. Not directly used for plotting,
                        but included for MATLAB function signature compatibility.
        num_conds (int): Number of conditions to plot.
        avg_data (dict): Dictionary containing average data, error data, and corrected p-values.
                         Keys are dynamically generated (e.g., 'ERPAVG_avg', 'ERPAVG_err', 'ERP_pvalues_corr').
        chan_idx (int): Index of the channel to plot.
        xaxis_indices (np.ndarray): Array of indices for the time points to plot.
        plot_conds (list): List of condition numbers.
        color_array (list): List of RGB color tuples/lists for each condition.
        line_thickness (float): Thickness of the mean lines.
        figure_name (str): Name of the figure, used for title and dynamic key generation.
        time_vector (np.ndarray): Full time vector corresponding to the data.
        yaxis_text (str): Label for the y-axis.
        specs (dict): Dictionary containing plotting specifications like 'GridLines' and 'Yaxis'.
        legend_array (list): List of strings for the plot legend.
    """
    plt.figure(figsize=(10, 6)) # Create a new figure for each plot call
    
    # Removes the '-line' present in non-ERP data matrices
    array_name = str(figure_name)
    if '-line' in array_name:
        array_name = array_name.replace('-line', '')

    h_lines = [] # To store line handles for the legend
    for cond_idx in range(num_conds):
        avg_key = f'{array_name}{plot_conds[cond_idx]}_avg'
        err_key = f'{array_name}{plot_conds[cond_idx]}_err'

        # Retrieve data for the current channel and x-axis range
        mean_data = avg_data[avg_key][chan_idx, xaxis_indices]
        error_data = avg_data[err_key][chan_idx, xaxis_indices]

        plot_x = time_vector[xaxis_indices]

        # Plot mean line
        line, = plt.plot(plot_x, mean_data, color=color_array[cond_idx], linewidth=line_thickness)
        h_lines.append(line)

        # Plot shaded error region
        plt.fill_between(plot_x, mean_data - error_data, mean_data + error_data,
                         color=color_array[cond_idx], alpha=0.25, edgecolor='none')

    # Draw horizontal (at y=0) and vertical (at x=0) dashed reference lines
    line_color_ref = [0.6, 0.6, 0.6] # MATLAB's [.5 .5 .5]+.1
    plt.axhline(0, color=line_color_ref, linestyle='--', linewidth=2)
    plt.axvline(0, color=line_color_ref, linestyle='--', linewidth=2)

    # Plot Statistical Significance
    significant_blocks_found = False
    pvalues_corr_key = f'{array_name}pvalues_corr'
    if pvalues_corr_key in avg_data:
        # Ensure pvalues_corr is handled as a 1D array for the given channel and x-axis
        # Assuming avg_data[pvalues_corr_key] is (num_channels, num_timepoints)
        significant_points = avg_data[pvalues_corr_key][chan_idx, xaxis_indices]
        
        # Find indices where significant_points is 1
        significant_indices = np.where(significant_points == 1)[0]

        if len(significant_indices) > 0:
            # Find the starts of contiguous blocks
            block_starts_relative_idx = significant_indices[np.insert(np.diff(significant_indices) != 1, 0, True)]
            # Find the ends of contiguous blocks
            block_ends_relative_idx = significant_indices[np.append(np.diff(significant_indices) != 1, True)]
            
            current_ylim = plt.gca().get_ylim()
            tempmin = current_ylim[0]
            patch_height = abs(tempmin * 0.5) # As per MATLAB code

            # Calculate y-coordinates in axes fraction (0 to 1) for axvspan
            y_bottom_frac = (tempmin - current_ylim[0]) / (current_ylim[1] - current_ylim[0])
            y_top_frac = (tempmin + patch_height - current_ylim[0]) / (current_ylim[1] - current_ylim[0])
            
            plot_x_axis = time_vector[xaxis_indices] # The actual time values for the x-axis

            for start_rel_idx, end_rel_idx in zip(block_starts_relative_idx, block_ends_relative_idx):
                x_start_time = plot_x_axis[start_rel_idx]
                x_end_time = plot_x_axis[end_rel_idx]
                plt.axvspan(x_start_time, x_end_time, ymin=y_bottom_frac, ymax=y_top_frac,
                            facecolor=[0.1, 0.1, 0.1], alpha=0.5, zorder=-1)
            significant_blocks_found = True

    # Add Grid Lines if specified
    if specs.get('GridLines', 0) > 0:
        current_ylim = plt.gca().get_ylim()
        temptime = time_vector[xaxis_indices]
        grid_step = specs['GridLines']
        grid_array = np.arange(temptime[0], temptime[-1] + grid_step, grid_step)
        for grid_val in grid_array:
            plt.axvline(grid_val, color=[0.7, 0.7, 0.7], linestyle='--', linewidth=1, zorder=-2)

    # Change Plot Settings
    plt.xlabel('Time (Seconds)')
    plt.ylabel(yaxis_text)

    # Set title, adding '***' if statistical significance was found
    if significant_blocks_found:
        plt.title(f"{figure_name}***", fontweight='bold')
    else:
        plt.title(figure_name, fontweight='bold')

    # Ensure the legend is only drawn if there are valid handles and labels
    if h_lines and legend_array:
        plt.legend(h_lines, legend_array, loc='upper left', frameon=False)

    if 'Yaxis' in specs and specs['Yaxis'] is not None and len(specs['Yaxis']) == 2:
        plt.ylim(specs['Yaxis'])
    
    plt.grid(False) # Turn off default grid as we're drawing custom ones
    plt.tight_layout()
    plt.show()


def plot_spectral_figure(
    num_conds, avg_data, chan_idx, xaxis_indices, plot_conds, figure_name,
    yaxis_text, legend_array, plot_specs, time_vector, frex, spect_plot_indices, loglin_freq
):
    """
    Replicates the plotting logic found in `ECOG_plotter_dep_figures_spectral_v3.m` MATLAB function.

    Args:
        num_conds (int): Number of conditions to plot.
        avg_data (dict): Dictionary containing average spectral data and corrected p-values.
                         Keys are dynamically generated (e.g., 'ERSPSPECT_spect', 'ERSPSPECT_spect_pvalues_corr').
        chan_idx (int): Index of the channel to plot.
        xaxis_indices (np.ndarray): Array of indices for the time points to plot.
        plot_conds (list): List of condition numbers.
        figure_name (str): Name of the figure, used for title and dynamic key generation (e.g., 'ERSP').
        yaxis_text (str): Label for the colorbar (e.g., 'Normalized Amplitude').
        legend_array (list): List of strings for condition legends.
        plot_specs (dict): Dictionary containing plotting layout specifications (e.g., 'rowcount', 'colcount', 'plotlocs').
        time_vector (np.ndarray): Full time vector corresponding to the data.
        frex (np.ndarray): Array of frequencies.
        spect_plot_indices (list): Indices for subplots in PLOT_Specs.plotlocs for spectral plots.
        loglin_freq (bool): True if frequencies should be log-scaled, False for linear.
    """

    # --- 1. Create data variables ---
    # freq x time
    # Extract relevant time points from the time vector
    plot_times = time_vector[xaxis_indices]
    
    input_data = {}
    for cond_idx in range(num_conds):
        # Dynamically generate key for spectral data (e.g., 'ERSP201_spect')
        spect_key = f'{figure_name}{int(plot_conds[cond_idx])}_spect'
        # Squeeze data for the specific channel, frequencies, and x-axis range
        # MATLAB: squeeze(AVG.(genvarname(char(strcat(figure_name,num2str(plot_conds(Condx)),'_spect'))))(chanx,xaxis,:));
        # Based on MATLAB's AVG structure: avg_data[spect_key] is (n_channels, n_timepoints_full, n_freqs)
        input_data[f'data{cond_idx + 1}'] = avg_data[spect_key][chan_idx, xaxis_indices, :]

    # Extract corrected p-values
    # MATLAB: squeeze(AVG.(genvarname(char(strcat(figure_name,'_spect_pvalues_corr'))))(chanx,:,xaxis))';
    pvalues_corr_key = f'{figure_name}_spect_pvalues_corr'
    if pvalues_corr_key in avg_data:
        # Assuming avg_data[pvalues_corr_key] is (n_channels, n_timepoints_full, n_freqs)
        input_data['stats'] = avg_data[pvalues_corr_key][chan_idx, xaxis_indices, :]
    else:
        input_data['stats'] = np.zeros((len(xaxis_indices), len(frex))) # No stats if key not found


    # --- 2. Determine Color Limits ---
    tempmax = []
    tempmin = []
    for cond_idx in range(num_conds):
        # Data extracted is (n_timepoints_plot, n_freqs)
        data_2d = input_data[f'data{cond_idx + 1}']
        tempmax.append(np.max(data_2d))
        tempmin.append(np.min(data_2d))

    # MATLAB: templim = max([tempmax abs(tempmin)]);
    templim = max(np.max(tempmax), np.max(np.abs(tempmin))) if tempmax and tempmin else 0

    # MATLAB: If templim==0, templim=.05;
    if templim == 0:
        templim = 0.05

    # --- 3. Plot Data ---
    # Create a single figure that can hold multiple subplots
    plt.figure(figsize=(12, 4 * num_conds)) # Adjust figure size dynamically

    for cond_idx in range(num_conds):
        # MATLAB: subplot(PLOT_Specs.rowcount,PLOT_Specs.colcount,PLOT_Specs.plotlocs(spect_plot_index(Condx),:))
        # plot_specs.plotlocs is 1-indexed in MATLAB. Adjust for 0-indexed Python.
        subplot_index = plot_specs['plotlocs'][spect_plot_indices[cond_idx] - 1]
        plt.subplot(plot_specs['rowcount'], plot_specs['colcount'], subplot_index)

        # Get spectral data for current condition (n_timepoints_plot, n_freqs)
        current_spectral_data = input_data[f'data{cond_idx + 1}']

        if figure_name == 'ITPA':
            # MATLAB comments suggest circular colorbar/image for ITPA, but then uses contourf with jet
            # For now, default to contourf, similar to other types.
            # If 'ITPA' needs specific circular colormap or `imagesc`, this part needs more detail.
            # Based on MATLAB, it seems to default to contourf in practice
            print("Warning: ITPA specific plotting in MATLAB was commented out or used contourf. Using contourf.")

        # Contour plot
        # MATLAB: contourf(time(xaxis),frex,input_data.(genvarname(['data',num2str(Condx)]))',100,'linecolor','none');
        # current_spectral_data is (n_timepoints_plot, n_freqs). For contourf (X, Y, Z), Z should be (len(Y), len(X)).
        # So, we need to transpose current_spectral_data to (n_freqs, n_timepoints_plot)
        plt.contourf(plot_times, frex, current_spectral_data.T, 100, cmap='jet', vmin=-templim, vmax=templim)

        # Logarithmic frequency scaling
        if loglin_freq:
            # MATLAB: set(gca,'yscale','log','ytick',logspace(log10(min(frex)),log10(max(frex)),freqscale),'yticklabel',...)
            plt.yscale('log')
            # Determine appropriate number of log-spaced ticks, similar to MATLAB's 'freqscale'
            freqscale = min(len(frex), 10) # Max 10 ticks for readability
            if freqscale > 1:
                log_ticks = np.logspace(np.log10(frex.min()), np.log10(frex.max()), freqscale)
                plt.yticks(log_ticks, [f'{t:.1f}' for t in log_ticks])


        # Title
        # Check if sig at any time/freq. Add *** if sig
        # MATLAB: if ~isempty(find(input_data.stats==1)) ... title([char(array_name), ': ', char(legend_array(Condx)),'***'],'FontWeight','bold');
        # input_data['stats'] is (n_timepoints_plot, n_freqs)
        has_significant_stats = np.any(input_data['stats'] == 1)
        title_text = f'{figure_name}: {legend_array[cond_idx]}'
        if has_significant_stats:
            title_text += '***'
        plt.title(title_text, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar()
        cbar.set_label(yaxis_text)

        # Plot stats overlay
        # MATLAB: if ~isempty(find(input_data.stats==1)) ... contour(time(xaxis),frex,input_data.stats',1,'Color','k');
        if has_significant_stats:
            # input_data['stats'] is (n_timepoints_plot, n_freqs)
            # Use contour to draw outlines of significant regions, transposing for (n_freqs, n_timepoints_plot)
            plt.contour(plot_times, frex, input_data['stats'].T, levels=[0.5], colors='k', linestyles='-', linewidths=1.5)

        # Labels
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Frequency')

    plt.tight_layout() # Adjust subplot params for a tight layout
    plt.show()


def compute_parametric_statistics(
    input_data, p_val, errbartype, multcomp_type, n_perm,
    analysis_windows, tail
):
    """
    Replicates the parametric statistical testing and multiple comparison correction
    logic found in `ECOG_plotter_dep_parametric_statistics_v9.m` MATLAB function.

    Args:
        input_data (dict): A dictionary where keys are 'data1', 'data2', etc.,
                           each containing trial-level data (n_trials, n_timepoints, n_frequencies).
                           If n_frequencies is 1 (line data), the last dimension can be omitted.
        p_val (float): The p-value threshold for statistical significance (e.g., 0.05).
        errbartype (int): Type of error bar to compute:
                          1 for Standard Error of the Mean (SEM),
                          2 for One-Sample Confidence Interval.
        multcomp_type (int): Type of multiple comparison correction to apply:
                             0: None (apply p-value threshold directly).
                             1: False Discovery Rate (FDR).
                             2: MaxStat permutation test.
                             3: Cluster-based permutation test.
                             4: Bonferroni correction.
                             5: Mean Amplitude within time windows (Bonferroni corrected).
        n_perm (int): Number of permutations for permutation tests (if multcomp_type is 2 or 3).
        analysis_windows (dict): A dictionary where keys are 'TIMES1', 'TIMES2', etc.,
                                 each containing an array of time indices for a window of interest.
        tail (str): Type of test tail: 'both', 'Positive', or 'Negative'.

    Returns:
        dict: A dictionary containing the statistical results, mimicking MATLAB's STATS structure.
              Expected keys: 'p', 'tstat', 'ci', 'multcomp', 'multcompvals', 'perm' (if applicable).
    """

    # --- Initialization and Setup ---
    # Determine number of frequencies and conditions from input_data
    # Assuming input_data.data1 exists and has shape (n_trials, n_timepoints, n_frequencies) or (n_trials, n_timepoints)
    first_data_key = next(iter(input_data))
    data_shape = input_data[first_data_key].shape
    n_trials_first_cond = data_shape[0]
    n_timepoints = data_shape[1]
    num_freqs = data_shape[2] if len(data_shape) == 3 else 1
    n_conds = len(input_data)

    stats_results = {
        'p': np.full((num_freqs, n_timepoints), np.nan),
        'tstat': np.full((num_freqs, n_timepoints), np.nan),
        'ci': np.full((n_conds, n_timepoints), np.nan), # Adjusted to handle line data (num_freqs=1)
        'multcomp': np.zeros((num_freqs, n_timepoints)),
        'multcompvals': np.full((num_freqs, n_timepoints), np.nan)
    }

    # Combine all analysis windows into a single array
    analysis_windows_all = np.concatenate(list(analysis_windows.values())) if analysis_windows else np.arange(n_timepoints)
    # Create a boolean mask for easy indexing within analysis windows
    analysis_windows_all_bool = np.zeros(n_timepoints, dtype=bool)
    if analysis_windows_all.size > 0:
        analysis_windows_all_bool[analysis_windows_all] = True

    # --- Permutation Sufficiency Check ---
    if multcomp_type in [2, 3]:
        if n_perm < 1000:
            print("Warning: Number of permutations might be insufficient for robust results. Recommended at least 1000.")
        if tail != 'both':
            print("Warning: Permutation tests are primarily designed for two-tailed comparisons. Proceeding with one-tailed analysis.")

    # --- Compute Real Statistics ---
    if p_val > 0:
        if n_conds == 1:
            for freq_idx in range(num_freqs):
                data_slice = input_data[first_data_key][:, :, freq_idx] if num_freqs > 1 else input_data[first_data_key]
                t_stats, p_values = stats.ttest_1samp(data_slice, 0, axis=0, nan_policy='omit', alternative={'both': 'two-sided', 'Positive': 'greater', 'Negative': 'less'}[tail])
                stats_results['p'][freq_idx, :] = p_values
                stats_results['tstat'][freq_idx, :] = t_stats
        elif n_conds == 2:
            data_cond1 = input_data['data1']
            data_cond2 = input_data['data2']
            for freq_idx in range(num_freqs):
                data1_slice = data_cond1[:, :, freq_idx] if num_freqs > 1 else data_cond1
                data2_slice = data_cond2[:, :, freq_idx] if num_freqs > 1 else data_cond2
                t_stats, p_values = stats.ttest_ind(data1_slice, data2_slice, axis=0, nan_policy='omit', equal_var=True, alternative={'both': 'two-sided', 'Positive': 'greater', 'Negative': 'less'}[tail])
                stats_results['p'][freq_idx, :] = p_values
                stats_results['tstat'][freq_idx, :] = t_stats
        elif n_conds >= 3:
            for freq_idx in range(num_freqs):
                current_freq_data = []
                for data_key in input_data:
                    if num_freqs > 1:
                        current_freq_data.append(input_data[data_key][:, :, freq_idx])
                    else:
                        current_freq_data.append(input_data[data_key])

                f_stats = np.full(n_timepoints, np.nan)
                p_values = np.full(n_timepoints, np.nan)

                for t_idx in range(n_timepoints):
                    data_at_timepoint = [cond_data[:, t_idx] for cond_data in current_freq_data]
                    valid_data_at_timepoint = [arr[~np.isnan(arr)] for arr in data_at_timepoint if np.sum(~np.isnan(arr)) >= 2]

                    if len(valid_data_at_timepoint) >= 2: # f_oneway requires at least two groups
                        f_stat, p_val_anova = stats.f_oneway(*valid_data_at_timepoint)
                        f_stats[t_idx] = f_stat
                        p_values[t_idx] = p_val_anova

                stats_results['p'][freq_idx, :] = p_values
                stats_results['tstat'][freq_idx, :] = f_stats
    else:
        pass

    # --- Compute Confidence Intervals ---
    if num_freqs == 1: # Only for line stats
        for cond_idx, data_key in enumerate(input_data.keys()):
            current_data = input_data[data_key]
            if errbartype == 1: # SEM
                stats_results['ci'][cond_idx, :] = np.nanstd(current_data, axis=0) / np.sqrt(np.sum(~np.isnan(current_data), axis=0))
            elif errbartype == 2: # One-Sample Conf Int (simplified for example)
                # This is a simplified calculation; a full t-distribution based CI would be more accurate
                sem = np.nanstd(current_data, axis=0) / np.sqrt(np.sum(~np.isnan(current_data), axis=0))
                stats_results['ci'][cond_idx, :] = sem * 1.96 # Approx. for example

    # --- Multiple Comparisons Correction ---
    if p_val > 0:
        # Ensure p_values_to_correct_slice is always 2D (num_freqs, num_timepoints_in_slice)
        p_values_to_correct_slice = stats_results['p'][:, analysis_windows_all_bool]
        if p_values_to_correct_slice.ndim == 1: # Handle case where num_freqs == 1 after slicing
            p_values_to_correct_slice = p_values_to_correct_slice[np.newaxis, :]
        
        real_t_stats_slice = stats_results['tstat'][:, analysis_windows_all_bool]
        if real_t_stats_slice.ndim == 1:
            real_t_stats_slice = real_t_stats_slice[np.newaxis, :]

        if multcomp_type == 0: # None
            significant_mask = p_values_to_correct_slice < p_val
            stats_results['multcomp'][:, analysis_windows_all_bool] = significant_mask
            stats_results['multcompvals'][:, analysis_windows_all_bool] = p_values_to_correct_slice

        elif multcomp_type == 1: # FDR
            # Use the newly defined fdr_bh function
            h_flat, _, _, adj_p_flat = fdr_bh(p_values_to_correct_slice.flatten(), q=p_val, method='pdep')
            
            stats_results['multcomp'][:, analysis_windows_all_bool] = h_flat.reshape(p_values_to_correct_slice.shape)
            stats_results['multcompvals'][:, analysis_windows_all_bool] = adj_p_flat.reshape(p_values_to_correct_slice.shape)

        elif multcomp_type == 2: # MaxStat Permutation Test
            perm_tmax = np.zeros(n_perm)
            perm_tmin = np.zeros(n_perm)

            all_cond_data_list = [input_data[key] for key in sorted(input_data.keys())] # Preserve order
            n_trials_per_cond = [data.shape[0] for data in all_cond_data_list]

            for perm_x in range(n_perm):
                perm_input_data = {}
                if n_conds == 1:
                    # One-sample permutation using Sign-Switching
                    rand_signs = np.random.choice([-1, 1], size=n_trials_per_cond[0])
                    if num_freqs > 1:
                        perm_input_data[first_data_key] = all_cond_data_list[0] * rand_signs[:, np.newaxis, np.newaxis]
                    else: # Line data (n_trials, n_timepoints)
                        perm_input_data[first_data_key] = all_cond_data_list[0] * rand_signs[:, np.newaxis]

                    # Re-run one-sample t-test on permuted data
                    perm_tstat = np.full((num_freqs, n_timepoints), np.nan)
                    for freq_idx in range(num_freqs):
                        data_slice = perm_input_data[first_data_key][:, :, freq_idx] if num_freqs > 1 else perm_input_data[first_data_key]
                        t_stat_perm, _ = stats.ttest_1samp(data_slice, 0, axis=0, nan_policy='omit', alternative='two-sided')
                        perm_tstat[freq_idx, :] = t_stat_perm

                elif n_conds >= 2: # Two-sample or N-condition permutation (label shuffling)
                    total_trials = sum(n_trials_per_cond)
                    combined_data = np.concatenate(all_cond_data_list, axis=0)

                    shuffled_indices = np.random.permutation(total_trials)
                    shuffled_combined_data = combined_data[shuffled_indices]

                    # Re-split into permuted conditions
                    current_idx = 0
                    for k, n_trials in enumerate(n_trials_per_cond):
                        perm_input_data[f'data{k+1}'] = shuffled_combined_data[current_idx : current_idx + n_trials]
                        current_idx += n_trials

                    # Re-run appropriate test (t-test_ind or f_oneway)
                    perm_tstat = np.full((num_freqs, n_timepoints), np.nan)
                    for freq_idx in range(num_freqs):
                        if n_conds == 2:
                            data1_slice_perm = perm_input_data['data1'][:, :, freq_idx] if num_freqs > 1 else perm_input_data['data1']
                            data2_slice_perm = perm_input_data['data2'][:, :, freq_idx] if num_freqs > 1 else perm_input_data['data2']
                            t_stat_perm, _ = stats.ttest_ind(data1_slice_perm, data2_slice_perm, axis=0, nan_policy='omit', equal_var=True, alternative='two-sided')
                            perm_tstat[freq_idx, :] = t_stat_perm
                        elif n_conds >= 3:
                            current_freq_data_perm = []
                            for data_key_perm in perm_input_data:
                                if num_freqs > 1:
                                    current_freq_data_perm.append(perm_input_data[data_key_perm][:, :, freq_idx])
                                else:
                                    current_freq_data_perm.append(perm_input_data[data_key_perm])

                            f_stats_perm = np.full(n_timepoints, np.nan)
                            for t_idx in range(n_timepoints):
                                data_at_timepoint_perm = [cond_data[:, t_idx] for cond_data in current_freq_data_perm]
                                valid_data_at_timepoint_perm = [arr[~np.isnan(arr)] for arr in data_at_timepoint_perm if np.sum(~np.isnan(arr)) >= 2]

                                if len(valid_data_at_timepoint_perm) >= 2:
                                    f_stat_perm, _ = stats.f_oneway(*valid_data_at_timepoint_perm)
                                    f_stats_perm[t_idx] = f_stat_perm
                            perm_tstat[freq_idx, :] = f_stats_perm
                else:
                    raise ValueError("Unhandled n_conds for permutation test.")

                # Store full permuted tstat and record global max/min within analysis window
                perm_stats_slice = perm_tstat[:, analysis_windows_all_bool]
                if perm_stats_slice.ndim == 1:
                    perm_stats_slice = perm_stats_slice[np.newaxis, :]

                perm_tmax[perm_x] = np.nanmax(perm_stats_slice) if perm_stats_slice.size > 0 else np.nan
                perm_tmin[perm_x] = np.nanmin(perm_stats_slice) if perm_stats_slice.size > 0 else np.nan

            # Remove NaNs from perm_tmax/min if any (can happen if analysis_windows_all_bool is empty or slices are all NaNs)
            perm_tmax = perm_tmax[~np.isnan(perm_tmax)]
            perm_tmin = perm_tmin[~np.isnan(perm_tmin)]

            if perm_tmax.size == 0 or perm_tmin.size == 0:
                print("Warning: No valid statistics from permutations to calculate thresholds. Skipping MaxStat.")
                stats_results['multcomp'][:, analysis_windows_all_bool] = 0
                stats_results['multcompvals'][:, analysis_windows_all_bool] = np.nan
            else:
                # Determine critical thresholds
                perm_tmax.sort()
                perm_tmin.sort() # For negative effects

                alpha_lower_perc = (p_val / 2.0) * 100
                alpha_upper_perc = (1.0 - (p_val / 2.0)) * 100

                tmax_thresh = np.percentile(perm_tmax, alpha_upper_perc)
                tmin_thresh = np.percentile(perm_tmin, alpha_lower_perc)

                stats_results['perm'] = {
                    'tmaxThresh': tmax_thresh,
                    'tminThresh': tmin_thresh
                }

                # Apply thresholds to real t-statistics (within analysis window)
                significant_mask_max = real_t_stats_slice > tmax_thresh
                significant_mask_min = real_t_stats_slice < tmin_thresh
                stats_results['multcomp'][:, analysis_windows_all_bool] = significant_mask_max | significant_mask_min

                # Calculate multcompvals (adjusted p-values) for MaxStat
                adjusted_p_values = np.full(real_t_stats_slice.shape, np.nan)
                # This is a simplified calculation, a more rigorous one would compute an empirical p-value for each pixel
                # based on its position in the sorted max/min distributions
                for f_idx in range(num_freqs):
                    for t_idx_slice in range(real_t_stats_slice.shape[1]):
                        real_t = real_t_stats_slice[f_idx, t_idx_slice]
                        if np.isnan(real_t):
                            continue

                        if real_t > 0:
                            p_val_adj = np.sum(perm_tmax >= real_t) / n_perm
                        else:
                            p_val_adj = np.sum(perm_tmin <= real_t) / n_perm
                        adjusted_p_values[f_idx, t_idx_slice] = p_val_adj * 2 if tail == 'both' else p_val_adj # Double for two-sided
                adjusted_p_values = np.minimum(adjusted_p_values, 1.0) # Cap at 1.0
                stats_results['multcompvals'][:, analysis_windows_all_bool] = adjusted_p_values

        elif multcomp_type == 3: # Cluster-based Permutation
            perm_max_cluster_sums_pos = np.zeros(n_perm)
            perm_max_cluster_sums_neg = np.zeros(n_perm)

            all_cond_data_list = [input_data[key] for key in sorted(input_data.keys())] # Preserve order
            n_trials_per_cond = [data.shape[0] for data in all_cond_data_list]

            # Determine if the input data is conceptually 1D (num_freqs == 1)
            is_input_1d_conceptually = (num_freqs == 1)

            # Connectivity structure for clustering
            if is_input_1d_conceptually:
                # For 1 frequency, it's a 1D problem along time. Squeeze to 1D and use 1D connectivity.
                connectivity_structure = generate_binary_structure(1, 1) # 2-connectivity for 1D (pixels connect horizontally)
            else:
                # For >1 frequency, it's a 2D problem (time x frequency).
                connectivity_structure = generate_binary_structure(2, 1) # 4-connectivity for 2D (horizontally or vertically)

            # --- Helper function to find clusters and sum t-stats ---
            def get_cluster_sums(t_map, p_map, alpha_threshold, conn_struct, is_1d_data):
                significant_pixels = (p_map < alpha_threshold) # Binary mask of significant pixels

                # If conceptually 1D (num_freqs = 1), squeeze the first dimension for `label`
                if is_1d_data:
                    significant_pixels_for_label = significant_pixels.squeeze()
                    t_map_for_label = t_map.squeeze()
                else:
                    significant_pixels_for_label = significant_pixels
                    t_map_for_label = t_map

                # Check if there are any significant pixels to cluster
                if not np.any(significant_pixels_for_label):
                    labeled_array_result = np.zeros_like(significant_pixels_for_label, dtype=int)
                    if is_1d_data: # If originally (1, N), expand back to (1, N)
                        labeled_array_result = labeled_array_result[np.newaxis, :]
                    return [], [], labeled_array_result

                labeled_array, num_features = label(significant_pixels_for_label, structure=conn_struct)

                cluster_sums_pos = []
                cluster_sums_neg = []

                for i in range(1, num_features + 1):
                    cluster_t_values = t_map_for_label[labeled_array == i] # Get t-values for current cluster

                    if np.sum(cluster_t_values) > 0:
                        cluster_sums_pos.append(np.sum(cluster_t_values))
                    else:
                        cluster_sums_neg.append(np.sum(cluster_t_values))

                # Ensure the returned labeled_array has the same dimensionality as the original t_map/p_map
                if is_1d_data:
                    return cluster_sums_pos, cluster_sums_neg, labeled_array[np.newaxis, :] # Expand back to (1, N)
                else:
                    return cluster_sums_pos, cluster_sums_neg, labeled_array

            # --- Real Data Clustering ---
            real_t_map_window = stats_results['tstat'][:, analysis_windows_all_bool]
            real_p_map_window = stats_results['p'][:, analysis_windows_all_bool]

            real_cluster_sums_pos, real_cluster_sums_neg, real_labeled_array_window = get_cluster_sums(
                real_t_map_window, real_p_map_window, p_val, connectivity_structure, is_input_1d_conceptually
            )

            # --- Permutation Loop for Cluster-based Test ---
            for perm_x in range(n_perm):
                perm_input_data = {}
                if n_conds == 1:
                    rand_signs = np.random.choice([-1, 1], size=n_trials_per_cond[0])
                    if num_freqs > 1:
                        perm_input_data[first_data_key] = all_cond_data_list[0] * rand_signs[:, np.newaxis, np.newaxis]
                    else:
                        perm_input_data[first_data_key] = all_cond_data_list[0] * rand_signs[:, np.newaxis]

                    perm_tstat = np.full((num_freqs, n_timepoints), np.nan)
                    perm_p = np.full((num_freqs, n_timepoints), np.nan)
                    for freq_idx in range(num_freqs):
                        data_slice = perm_input_data[first_data_key][:, :, freq_idx] if num_freqs > 1 else perm_input_data[first_data_key]
                        t_stat_perm, p_val_perm = stats.ttest_1samp(data_slice, 0, axis=0, nan_policy='omit', alternative='two-sided')
                        perm_tstat[freq_idx, :] = t_stat_perm
                        perm_p[freq_idx, :] = p_val_perm

                elif n_conds >= 2:
                    total_trials = sum(n_trials_per_cond)
                    combined_data = np.concatenate(all_cond_data_list, axis=0)

                    shuffled_indices = np.random.permutation(total_trials)
                    shuffled_combined_data = combined_data[shuffled_indices]

                    current_idx = 0
                    for k, n_trials in enumerate(n_trials_per_cond):
                        perm_input_data[f'data{k+1}'] = shuffled_combined_data[current_idx : current_idx + n_trials]
                        current_idx += n_trials

                    perm_tstat = np.full((num_freqs, n_timepoints), np.nan)
                    perm_p = np.full((num_freqs, n_timepoints), np.nan)
                    for freq_idx in range(num_freqs):
                        if n_conds == 2:
                            data1_slice_perm = perm_input_data['data1'][:, :, freq_idx] if num_freqs > 1 else perm_input_data['data1']
                            data2_slice_perm = perm_input_data['data2'][:, :, freq_idx] if num_freqs > 1 else perm_input_data['data2']
                            t_stat_perm, p_val_perm = stats.ttest_ind(data1_slice_perm, data2_slice_perm, axis=0, nan_policy='omit', equal_var=True, alternative='two-sided')
                            perm_tstat[freq_idx, :] = t_stat_perm
                            perm_p[freq_idx, :] = p_val_perm
                        elif n_conds >= 3:
                            current_freq_data_perm = []
                            for data_key_perm in perm_input_data:
                                if num_freqs > 1:
                                    current_freq_data_perm.append(perm_input_data[data_key_perm][:, :, freq_idx])
                                else:
                                    current_freq_data_perm.append(perm_input_data[data_key_perm])

                            f_stats_perm = np.full(n_timepoints, np.nan)
                            p_values_perm = np.full(n_timepoints, np.nan)
                            for t_idx in range(n_timepoints):
                                data_at_timepoint_perm = [cond_data[:, t_idx] for cond_data in current_freq_data_perm]
                                valid_data_at_timepoint_perm = [arr[~np.isnan(arr)] for arr in data_at_timepoint_perm if np.sum(~np.isnan(arr)) >= 2]

                                if len(valid_data_at_timepoint_perm) >= 2:
                                    f_stat_perm, p_val_anova_perm = stats.f_oneway(*valid_data_at_timepoint_perm)
                                    f_stats_perm[t_idx] = f_stat_perm
                                    p_values_perm[t_idx] = p_val_anova_perm
                            perm_tstat[freq_idx, :] = f_stats_perm
                            perm_p[freq_idx, :] = p_values_perm
                else:
                    raise ValueError("Unhandled n_conds for permutation test.")

                # Get permuted cluster sums for the current analysis window
                perm_t_map_window = perm_tstat[:, analysis_windows_all_bool]
                perm_p_map_window = perm_p[:, analysis_windows_all_bool]

                perm_cluster_sums_pos, perm_cluster_sums_neg, _ = get_cluster_sums(
                    perm_t_map_window, perm_p_map_window, p_val, connectivity_structure, is_input_1d_conceptually
                )

                perm_max_cluster_sums_pos[perm_x] = np.max(perm_cluster_sums_pos) if perm_cluster_sums_pos else 0
                perm_max_cluster_sums_neg[perm_x] = np.min(perm_cluster_sums_neg) if perm_cluster_sums_neg else 0

            # --- Determine Critical Thresholds from Permuted Distribution ---
            perm_max_cluster_sums_pos.sort() # Sort ascending for upper threshold
            perm_max_cluster_sums_neg.sort() # Sort ascending for lower threshold (most negative is at the beginning)

            # Ensure we don't try to index beyond the array size for alpha_upper_idx and alpha_lower_idx
            # if n_perm * p_val / 2.0 results in 0 (e.g. for very small n_perm or p_val)
            alpha_idx_lower = int(np.floor(n_perm * (p_val / 2.0)))
            alpha_idx_upper = int(np.ceil(n_perm * (1 - p_val / 2.0))) - 1

            if alpha_idx_upper >= n_perm: alpha_idx_upper = n_perm - 1
            if alpha_idx_lower < 0: alpha_idx_lower = 0
            if alpha_idx_lower >= n_perm: alpha_idx_lower = n_perm - 1 # Should not happen if n_perm is reasonable

            # Handle case where distributions might be empty after filtering 0s for cluster sums
            # If perm_max_cluster_sums_pos is empty, np.percentile will fail or give nan.
            # Set thresholds to inf/-inf or 0 appropriately.
            cluster_thresh_pos = 0 if perm_max_cluster_sums_pos.size == 0 else perm_max_cluster_sums_pos[alpha_idx_upper]
            cluster_thresh_neg = 0 if perm_max_cluster_sums_neg.size == 0 else perm_max_cluster_sums_neg[alpha_idx_lower]

            stats_results['perm'] = {
                'tmaxThresh': cluster_thresh_pos,
                'tminThresh': cluster_thresh_neg,
                'perm_max_cluster_sums_pos': perm_max_cluster_sums_pos,
                'perm_max_cluster_sums_neg': perm_max_cluster_sums_neg
            }

            # --- Apply Thresholds to Real Data Clusters ---
            multcomp_mask = np.zeros_like(real_t_map_window, dtype=bool)
            multcompvals_array = np.full_like(real_t_map_window, np.nan)

            if real_labeled_array_window.size > 0 and real_labeled_array_window.max() > 0: # Check if any clusters found in real data
                for i in range(1, real_labeled_array_window.max() + 1): # Iterate through real clusters
                    cluster_pixels_mask = (real_labeled_array_window == i)
                    cluster_t_values = real_t_map_window[cluster_pixels_mask]
                    current_cluster_sum = np.sum(cluster_t_values)

                    is_significant = False
                    adjusted_p = np.nan

                    if current_cluster_sum > 0 and current_cluster_sum >= cluster_thresh_pos:
                        is_significant = True
                        # Approximate p-value for positive clusters
                        if perm_max_cluster_sums_pos.size > 0:
                            adjusted_p = np.sum(perm_max_cluster_sums_pos >= current_cluster_sum) / n_perm
                    elif current_cluster_sum < 0 and current_cluster_sum <= cluster_thresh_neg:
                        is_significant = True
                        # Approximate p-value for negative clusters
                        if perm_max_cluster_sums_neg.size > 0:
                            adjusted_p = np.sum(perm_max_cluster_sums_neg <= current_cluster_sum) / n_perm

                    if is_significant:
                        multcomp_mask[cluster_pixels_mask] = True
                        multcompvals_array[cluster_pixels_mask] = adjusted_p

            stats_results['multcomp'][:, analysis_windows_all_bool] = multcomp_mask
            stats_results['multcompvals'][:, analysis_windows_all_bool] = multcompvals_array

        elif multcomp_type == 4: # Bonferroni correction
            num_comparisons = p_values_to_correct_slice.size
            p_bonferroni_corrected = p_values_to_correct_slice * num_comparisons
            p_bonferroni_corrected[p_bonferroni_corrected > 1.0] = 1.0
            significant_mask = p_bonferroni_corrected < p_val
            stats_results['multcomp'][:, analysis_windows_all_bool] = significant_mask
            stats_results['multcompvals'][:, analysis_windows_all_bool] = p_bonferroni_corrected

        elif multcomp_type == 5: # Time Windows, Mean Amplitude
            print("Warning: Mean Amplitude window stats not fully implemented in this placeholder.")
            pass

    # --- Sanity Check (simplified) ---
    # In a full implementation, this would check consistency between 'multcomp' and 'multcompvals'

    return stats_results


def fdr_bh(pvals, q=0.05, method='pdep', report='no'):
    """
    Replicates the Benjamini & Hochberg (1995) and Benjamini & Yekutieli (2001)
    procedure for controlling the false discovery rate (FDR).
    Adapted from fdr_bh.m MATLAB function.

    Args:
        pvals (np.ndarray): A vector or matrix containing the p-value of each individual test.
        q (float): The desired false discovery rate. (default: 0.05)
        method (str): 'pdep' for independence or positive dependence (BH procedure),
                      'dep' for any dependency structure (BY procedure). (default: 'pdep')
        report (str): 'yes' or 'no' to print a brief summary. (default: 'no')

    Returns:
        tuple: (h, crit_p, adj_ci_cvrg, adj_p)
            h (np.ndarray): Binary array of same size as pvals. 1 if significant, 0 otherwise.
            crit_p (float): All uncorrected p-values <= crit_p are significant. 0 if no p-values significant.
            adj_ci_cvrg (float): The FCR-adjusted confidence interval coverage. NaN if no p-values significant.
            adj_p (np.ndarray): Adjusted p-values, same size as pvals. Can be > 1.
    """

    # Input validation
    if not isinstance(pvals, np.ndarray):
        pvals = np.array(pvals)

    if np.any(pvals < 0) or np.any(pvals > 1):
        raise ValueError('p-values must be between 0 and 1.')

    original_shape = pvals.shape
    p_flat = pvals.flatten()
    m = len(p_flat) # number of tests

    # Sort p-values and keep track of original indices
    sort_indices = np.argsort(p_flat)
    p_sorted = p_flat[sort_indices]

    # Unsort indices to return results to original order
    unsort_indices = np.argsort(sort_indices)

    # Calculate thresholds and weighted p-values based on method
    if method.lower() == 'pdep':
        # BH procedure for independence or positive dependence
        rank = np.arange(1, m + 1)
        thresh = rank * q / m
        wtd_p = m * p_sorted / rank
    elif method.lower() == 'dep':
        # BH procedure for any dependency structure (Benjamini & Yekutieli, 2001)
        rank = np.arange(1, m + 1)
        denom = m * np.sum(1.0 / rank)
        thresh = rank * q / denom
        wtd_p = denom * p_sorted / rank
    else:
        raise ValueError("Argument 'method' needs to be 'pdep' or 'dep'.")

    # Compute adjusted p-values (adj_p)
    adj_p_sorted = np.zeros(m)
    adj_p_sorted[m - 1] = wtd_p[m - 1] # Start from the largest p-value (smallest rank)
    for k in range(m - 2, -1, -1):
        adj_p_sorted[k] = min(wtd_p[k], adj_p_sorted[k + 1])

    # Reshape adjusted p-values back to original shape
    adj_p = adj_p_sorted[unsort_indices].reshape(original_shape)
    adj_p[adj_p > 1.0] = 1.0 # Cap adjusted p-values at 1.0 (though MATLAB allows >1)

    # Determine significance (h) and critical p-value (crit_p)
    rej = p_sorted <= thresh
    max_id = np.where(rej)[0] # Find indices where p_sorted <= thresh

    if max_id.size == 0:
        crit_p = 0.0
        h = np.zeros(original_shape, dtype=bool)
        adj_ci_cvrg = np.nan
    else:
        crit_p = p_sorted[max_id[-1]] # Largest p-value that is significant
        h = (p_flat <= crit_p).reshape(original_shape) # Binary mask based on crit_p
        adj_ci_cvrg = 1 - thresh[max_id[-1]]

    # Report if requested
    if report.lower() == 'yes':
        n_sig = np.sum(h)
        if n_sig == 1:
            print(f"Out of {m} tests, {n_sig} is significant using a false discovery rate of {q}.")
        else:
            print(f"Out of {m} tests, {n_sig} are significant using a false discovery rate of {q}.")
        if method.lower() == 'pdep':
            print("FDR/FCR procedure used is guaranteed valid for independent or positively dependent tests.")
        else:
            print("FDR/FCR procedure used is guaranteed valid for independent or dependent tests.")

    return h, crit_p, adj_ci_cvrg, adj_p


def shaded_error_bar(x, y, err_bar, line_props=None, transparent=0):
    """
    Replicates the functionality of `shadedErrorBar.m` MATLAB function to create a 2-D line plot
    with a shaded error bar.

    Args:
        x (np.ndarray): A vector of x values. If empty, it will default to 1:length(y).
        y (np.ndarray): A vector of y values (mean line).
        err_bar (np.ndarray or list): Error bar values. Can be:
                                      - A 1D array: Symmetric error bars.
                                      - A 2D array of shape (2, len(x)): Asymmetric error bars
                                        (row 0 for upper, row 1 for lower).
                                      - A list of two callables: The first processes y for the main line,
                                        the second processes y for the error bar (not fully implemented as per MATLAB).
        line_props (dict or str): Dictionary of line properties for `plt.plot` (e.g., {'color': 'k', 'linestyle': '-'}) or
                                  a format string (e.g., '-k'). Defaults to '-k'.
        transparent (int): If 1, the shaded error bar is made transparent. (0 by default).

    Returns:
        dict: A dictionary of handles to the generated plot objects, including 'main_line' and 'patch'.
    """

    # Input processing similar to MATLAB
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    y = y.flatten() # Ensure y is a 1D array

    if x is None or len(x) == 0:
        x = np.arange(1, len(y) + 1)
    else:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x = x.flatten()

    if len(x) != len(y):
        raise ValueError('length(x) must equal length(y)')

    # Process err_bar
    if isinstance(err_bar, list) and all(callable(f) for f in err_bar): # Function handles case
        # This part of MATLAB logic is typically handled before calling the plotting function
        # For this translation, we assume y and err_bar are already processed or directly provided
        # For example, mean and std of multiple observations can be pre-calculated.
        pass # Assuming y and err_bar are already scalar values per x-point.
    elif isinstance(err_bar, np.ndarray):
        if err_bar.ndim == 1:
            # Symmetric error bars
            err_bar = np.vstack([err_bar, err_bar]) # Make it (2, len(x))
        elif err_bar.ndim == 2 and err_bar.shape[0] == 2:
            pass # Already (2, len(x))
        elif err_bar.ndim == 2 and err_bar.shape[1] == 2:
            err_bar = err_bar.T # Transpose if (len(x), 2)
        else:
            raise ValueError('err_bar has the wrong size or dimensions')
    else:
        raise TypeError('err_bar must be a numpy array or a list of callables')

    if len(x) != err_bar.shape[1]:
        raise ValueError('length(x) must equal the second dimension of err_bar')

    # Set default line properties
    if line_props is None:
        line_props = {'color': 'k', 'linestyle': '-'}
    elif isinstance(line_props, str):
        # Convert MATLAB-like string to matplotlib dict
        if '-' in line_props: line_props = {'color': line_props.replace('-', ''), 'linestyle': '-'}
        else: line_props = {'color': line_props}

    # Plot to get the parameters of the line (e.g., color)
    # In matplotlib, we get color directly from line_props or default
    if isinstance(line_props, dict):
        col = line_props.get('color', 'k')
    else:
        col = 'k' # Default if not dict

    # Work out the color of the shaded region
    patch_saturation = 0.15
    if transparent:
        face_alpha = patch_saturation
        patch_color = col
    else:
        face_alpha = 1
        # Adjust patch color for de-saturation, similar to MATLAB's col + (1-col)*(1-patchSaturation)
        if isinstance(col, str):
            # For named colors, matplotlib can convert, but simpler to use RGBA
            # This approximation is for demonstration; actual desaturation is more complex
            patch_color = col # Or choose a lighter shade manually
        else:
            # Assuming col is RGB tuple/list, apply desaturation logic
            patch_color = np.array(col) + (1 - np.array(col)) * (1 - patch_saturation)
            patch_color = np.clip(patch_color, 0, 1)

    # Calculate the error bar envelopes
    u_e = y + err_bar[0, :]
    l_e = y - err_bar[1, :]

    # Make the patch
    # Remove nans similar to MATLAB
    valid_indices = ~np.isnan(y) & ~np.isnan(u_e) & ~np.isnan(l_e)
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    u_e_valid = u_e[valid_indices]
    l_e_valid = l_e[valid_indices]

    x_patch = np.concatenate([x_valid, x_valid[::-1]])
    y_patch = np.concatenate([l_e_valid, u_e_valid[::-1]])

    h = {}
    h['patch'] = plt.fill(x_patch, y_patch, facecolor=patch_color, edgecolor='none', alpha=face_alpha)[0]

    # Make pretty edges around the patch.
    # MATLAB's edgeColor = col + (1-col)*0.55
    if isinstance(col, str):
        edge_color = 'gray' # Approximate for named colors
    else:
        edge_color = np.array(col) + (1 - np.array(col)) * 0.55
        edge_color = np.clip(edge_color, 0, 1)
    
    h['edge_upper'], = plt.plot(x_valid, u_e_valid, linestyle='-', color=edge_color)
    h['edge_lower'], = plt.plot(x_valid, l_e_valid, linestyle='-', color=edge_color)

    # Now plot the main line
    h['main_line'], = plt.plot(x, y, **line_props)

    return h


def suptitle(fig, text, fontsize=18, titleypos=0.95, plotregion=0.92):
    """
    Replicates the functionality of `suptitle.m` MATLAB function.
    Puts a title above all subplots, adjusting subplot positions if necessary.

    Args:
        fig (matplotlib.figure.Figure): The figure object to which the super title will be added.
        text (str): The main title text.
        fontsize (int): Font size for the super title. Defaults to 18.
        titleypos (float): Normalized y position of the title within the figure. Defaults to 0.95.
        plotregion (float): The maximum normalized y-extent that subplots should occupy.
                            Defaults to 0.92 (meaning 92% of the figure height from bottom).
    """

    renderer = fig.canvas.get_renderer()
    if renderer is None:
        # Draw the canvas once to get a renderer if it's not available
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    min_y = 1.0
    max_y = 0.0
    # Iterate through all axes to find the actual occupied y-extent
    for ax in fig.get_axes():
        bbox = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        min_y = min(min_y, bbox.y0)
        max_y = max(max_y, bbox.y1)

    # Calculate a scaling factor to adjust subplot positions if needed
    # If max_y (actual top of subplots) is above plotregion (desired max for subplots),
    # we need to scale down the subplots to fit below the suptitle.
    scale = 1.0
    if max_y > plotregion:
        # How much vertical space do we need to 'compress' the subplots into?
        # The available height for subplots is plotregion - min_y
        # The current height occupied by subplots is max_y - min_y
        scale = (plotregion - min_y) / (max_y - min_y)
        # Ensure scale is not less than a reasonable minimum to prevent tiny plots
        scale = max(scale, 0.5) # Arbitrary minimum scale to avoid squishing too much

    # Adjust the position of each subplot to make space
    if scale < 1.0:
        for ax in fig.get_axes():
            pos = ax.get_position()
            # Apply scaling relative to min_y, then shift up to align with original min_y
            new_y0 = (pos.y0 - min_y) * scale + min_y
            new_height = pos.height * scale
            ax.set_position([pos.x0, new_y0, pos.width, new_height])

    # Create a new invisible Axes object that spans the entire figure
    # This is similar to MATLAB's axes('pos',[0 1 1 1],'visible','off')
    supt_ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    supt_ax.set_visible(False)

    # Add the title text to this new Axes object
    text_obj = supt_ax.text(
        0.5, titleypos,
        text,
        ha='center', va='top',
        fontsize=fontsize, 
        fontweight='bold', # MATLAB 'FontWeight','bold'
        fontfamily='Times' # MATLAB 'FontName', 'Times'
    )

    # Ensure the title is always visible by managing its zorder if necessary
    text_obj.set_zorder(100) # Arbitrarily high zorder

    # Restore original renderer if it was changed
    # In matplotlib, this is generally not needed as renderer is managed by canvas.


def stats_one_cond(num_freqs, input_data_temp, p_val, tail):
    """
    Replicates the statistical testing logic found in `Stats_oneCond_v2.m`.
    Performs a one-sample t-test and calculates Cohen's D for a single condition.

    Args:
        num_freqs (int): Number of frequencies to process. If 1, assumes line data.
        input_data_temp (dict): Dictionary containing the data for one condition.
                                Expected key: 'data1' with shape (n_trials, n_timepoints, n_frequencies)
                                or (n_trials, n_timepoints) if n_frequencies is 1.
        p_val (float): The alpha level for the t-test (not directly used for p-value calculation here but
                       kept for MATLAB signature compatibility).
        tail (str): Type of test tail: 'both', 'Positive', or 'Negative'.

    Returns:
        dict: A dictionary containing 'p' (p-values), 'tstat' (t-statistics), and 'CohenD' (Cohen's D).
    """
    data1 = input_data_temp['data1']

    # Determine n_timepoints from the data shape
    if num_freqs == 1:
        # For line data, data1 shape is (n_trials, n_timepoints)
        n_timepoints = data1.shape[1]
    else:
        # For spectral data, data1 shape is (n_trials, n_timepoints, n_frequencies)
        n_timepoints = data1.shape[1]

    STATS_temp = {
        'p': np.full((num_freqs, n_timepoints), np.nan),
        'tstat': np.full((num_freqs, n_timepoints), np.nan),
        'CohenD': np.full((num_freqs, n_timepoints), np.nan)
    }

    for freqx in range(num_freqs):
        if num_freqs > 1:
            # Extract data slice for the current frequency: (n_trials, n_timepoints)
            data_slice = data1[:, :, freqx]
        else:
            # For line data, use the entire data1 as the single frequency slice
            data_slice = data1

        # Perform one-sample t-test against zero
        # nan_policy='omit' handles NaNs by excluding them, similar to MATLAB's ttest behavior
        # alternative parameter maps MATLAB's 'tail' options
        t_stats, p_values = stats.ttest_1samp(
            data_slice,
            popmean=0,
            axis=0,
            nan_policy='omit',
            alternative={'both': 'two-sided', 'Positive': 'greater', 'Negative': 'less'}[tail]
        )

        # Store results
        STATS_temp['p'][freqx, :] = p_values
        STATS_temp['tstat'][freqx, :] = t_stats

        # Calculate Cohen's D: mean / std. Ensure to handle division by zero.
        mean_data = np.nanmean(data_slice, axis=0)
        std_data = np.nanstd(data_slice, axis=0)

        # Avoid division by zero: if std_data is 0, CohenD is 0 (or NaN if mean is also 0)
        # Or, set to NaN where std is 0 to be conservative.
        cohen_d_values = np.full_like(mean_data, np.nan, dtype=float)
        non_zero_std_idx = std_data != 0
        cohen_d_values[non_zero_std_idx] = mean_data[non_zero_std_idx] / std_data[non_zero_std_idx]

        STATS_temp['CohenD'][freqx, :] = cohen_d_values

    return STATS_temp


def stats_two_cond(num_freqs, input_data_temp, p_val, tail):
    """
    Replicates the statistical testing logic found in `Stats_twoCond_v2.m`.
    Performs an independent two-sample t-test and calculates Cohen's D for two conditions.

    Args:
        num_freqs (int): Number of frequencies to process. If 1, assumes line data.
        input_data_temp (dict): Dictionary containing the data for two conditions.
                                Expected keys: 'data1' and 'data2', each with shape
                                (n_trials, n_timepoints, n_frequencies) or (n_trials, n_timepoints)
                                if n_frequencies is 1.
        p_val (float): The alpha level for the t-test (not directly used for p-value calculation here but
                       kept for MATLAB signature compatibility).
        tail (str): Type of test tail: 'both', 'Positive', or 'Negative'.

    Returns:
        dict: A dictionary containing 'p' (p-values), 'tstat' (t-statistics), and 'CohenD' (Cohen's D).
    """
    data1 = input_data_temp['data1']
    data2 = input_data_temp['data2']

    # Determine n_timepoints from the data shape (assuming data1 and data2 have consistent shapes)
    if num_freqs == 1:
        # For line data, shape is (n_trials, n_timepoints)
        n_timepoints = data1.shape[1]
    else:
        # For spectral data, shape is (n_trials, n_timepoints, n_frequencies)
        n_timepoints = data1.shape[1]

    STATS_temp = {
        'p': np.full((num_freqs, n_timepoints), np.nan),
        'tstat': np.full((num_freqs, n_timepoints), np.nan),
        'CohenD': np.full((num_freqs, n_timepoints), np.nan)
    }

    for freqx in range(num_freqs):
        if num_freqs > 1:
            # Extract data slice for the current frequency: (n_trials, n_timepoints)
            data1_slice = data1[:, :, freqx]
            data2_slice = data2[:, :, freqx]
        else:
            # For line data, use the entire data as the single frequency slice
            data1_slice = data1
            data2_slice = data2

        # Perform independent two-sample t-test
        # nan_policy='omit' handles NaNs by excluding them, similar to MATLAB's ttest2 behavior
        # alternative parameter maps MATLAB's 'Tail' options
        t_stats, p_values = stats.ttest_ind(
            data1_slice,
            data2_slice,
            axis=0,
            equal_var=True, # MATLAB's ttest2 defaults to equal variance, use True to match
            nan_policy='omit',
            alternative={'both': 'two-sided', 'Positive': 'greater', 'Negative': 'less'}[tail]
        )

        # Store results
        STATS_temp['p'][freqx, :] = p_values
        STATS_temp['tstat'][freqx, :] = t_stats

        # Calculate Cohen's D: (mean2 - mean1) / pooled_std
        mean1_data = np.nanmean(data1_slice, axis=0)
        mean2_data = np.nanmean(data2_slice, axis=0)
        
        # Calculate pooled standard deviation
        # Degrees of freedom for each group
        n1 = np.sum(~np.isnan(data1_slice), axis=0)
        n2 = np.sum(~np.isnan(data2_slice), axis=0)

        # Variance for each group
        var1 = np.nanvar(data1_slice, axis=0, ddof=1) # ddof=1 for sample variance
        var2 = np.nanvar(data2_slice, axis=0, ddof=1) # ddof=1 for sample variance

        pooled_std = np.full_like(mean1_data, np.nan, dtype=float)

        # Mask for valid pooled standard deviation calculation
        # Need at least 2 non-NaN samples in each group for variance, and total df > 0
        valid_mask = (n1 > 1) & (n2 > 1) & ((n1 + n2 - 2) > 0)
        
        if np.any(valid_mask):
            numerator = (n1[valid_mask] - 1) * var1[valid_mask] + \
                        (n2[valid_mask] - 1) * var2[valid_mask]
            denominator = n1[valid_mask] + n2[valid_mask] - 2
            pooled_std[valid_mask] = np.sqrt(numerator / denominator)

        cohen_d_values = (mean2_data - mean1_data) / pooled_std
        
        STATS_temp['CohenD'][freqx, :] = cohen_d_values

    return STATS_temp

