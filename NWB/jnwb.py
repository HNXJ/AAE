import uuid
import h5py
import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.ecephys import LFP, ElectricalSeries, ElectrodeGroup

from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5 import H5DataIO
from pynwb.file import Subject
from pynwb.misc import Units

import os
import matplotlib.pyplot as plt
import scipy.stats # Added missing import
import scipy.ndimage as ndimage # Added missing import

import scipy.signal as signal
import matplotlib.patches as patches
from matplotlib.colors import Normalize


def inspect_h5py_raw_structure(filepath, max_display_elements=5):
    """
    Recursively inspects a raw HDF5 file (e.g., an NWB file) using h5py,
    printing its group and dataset structure.

    Args:
        filepath (str): The path to the HDF5 file.
        max_display_elements (int): Maximum number of elements to display for small arrays.
    """

    def _print_item(name, obj, indent=0):
        indent_str = '  ' * indent
        if isinstance(obj, h5py.Group):
            print(f"{indent_str}Group: {name}/")
            for key, val in obj.items():
                _print_item(key, val, indent + 1)
        elif isinstance(obj, h5py.Dataset):
            value_info = f"Shape: {obj.shape}, Dtype: {obj.dtype}"
            if obj.size <= max_display_elements and obj.ndim <= 1: # Only display small 1D datasets directly
                try:
                    value_info += f", Value: {obj[()]}"
                except Exception as e:
                    value_info += f" (Error reading value: {e})"
            print(f"{indent_str}Dataset: {name} ({value_info})")
        else:
            print(f"{indent_str}Unknown: {name} (Type: {type(obj).__name__})")

    print(f"\n--- Inspecting Raw HDF5 Structure of: {filepath} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            _print_item(f.name, f, indent=0)
    except Exception as e:
        print(f"Error accessing HDF5 file: {e}")
    print("--- End Raw HDF5 Inspection ---")


def reconstruct_nwb_inspected(source_filepath, target_filepath):
    """
    Inspects the raw HDF5 structure and simultaneously reconstructs a valid NWBFile.
    Attempts to preserve the structure of acquisition, processing, stimulus, and metadata.
    """
    print(f"--- Reconstructing with Inspection: {source_filepath} -> {target_filepath} ---")

    def _safe_decode(val):
        if val is None: return None
        if isinstance(val, (bytes, np.bytes_)):
            return val.decode('utf-8')
        if isinstance(val, np.ndarray):
            if val.ndim == 0: return _safe_decode(val.item())
            return [_safe_decode(v) for v in val]
        return str(val)

    with h5py.File(source_filepath, 'r') as f:
        # 1. General Metadata
        print("Inspecting: / (Root Metadata)")
        sst_str = _safe_decode(f.get('session_start_time')[()]) if 'session_start_time' in f else None
        try:
            clean_str = sst_str.strip("b'").strip('"')
            session_start_time = datetime.fromisoformat(clean_str)
        except:
            session_start_time = datetime.now(tzlocal())

        identifier = str(uuid.uuid4())
        desc = _safe_decode(f.get('general/experiment_description')[()]) if 'general/experiment_description' in f else "Reconstructed"
        session_id = _safe_decode(f.get('general/session_id')[()]) if 'general/session_id' in f else None

        metadata_fields = {
            'notes': f.get('general/notes'),
            'pharmacology': f.get('general/pharmacology'),
            'protocol': f.get('general/protocol'),
            'surgery': f.get('general/surgery'),
            'virus': f.get('general/virus'),
            'slices': f.get('general/slices'),
            'data_collection': f.get('general/data_collection'),
            'stimulus_notes': f.get('general/stimulus')
        }

        nwb_kwargs = {}
        for key, dset in metadata_fields.items():
            if dset is not None:
                val = _safe_decode(dset[()])
                if val:
                    nwb_kwargs[key] = val

        subject = None
        if 'general/subject' in f:
            print("  - Found Subject metadata")
            subj_grp = f['general/subject']
            subject = Subject(
                subject_id=_safe_decode(subj_grp.get('subject_id')[()]) if 'subject_id' in subj_grp else 'unknown',
                description=_safe_decode(subj_grp.get('description')[()]) if 'description' in subj_grp else None,
                species=_safe_decode(subj_grp.get('species')[()]) if 'species' in subj_grp else None,
                sex=_safe_decode(subj_grp.get('sex')[()]) if 'sex' in subj_grp else None,
                age=_safe_decode(subj_grp.get('age')[()]) if 'age' in subj_grp else None
            )

        nwbfile = NWBFile(
            session_description=desc,
            identifier=identifier,
            session_start_time=session_start_time,
            institution=_safe_decode(f.get('general/institution')[()]) if 'general/institution' in f else None,
            lab=_safe_decode(f.get('general/lab')[()]) if 'general/lab' in f else None,
            experimenter=_safe_decode(f.get('general/experimenter')[()]) if 'general/experimenter' in f else None,
            session_id=session_id,
            subject=subject,
            **nwb_kwargs
        )

        # 2. Devices & Electrodes
        print("Inspecting: /general/devices & /general/extracellular_ephys")
        device_map = {}
        if 'general/devices' in f:
            for dev_name in f['general/devices']:
                device_map[dev_name] = nwbfile.create_device(name=dev_name)
        if not device_map: device_map['default'] = nwbfile.create_device(name='default_device')

        eg_map = {}
        if 'general/extracellular_ephys' in f:
            ephys = f['general/extracellular_ephys']
            for key in ephys:
                if key == 'electrodes': continue
                dev = list(device_map.values())[0]
                eg_map[key] = nwbfile.create_electrode_group(name=key, description=key, location="unknown", device=dev)

        if 'general/extracellular_ephys/electrodes' in f:
            elec_dset = f['general/extracellular_ephys/electrodes']
            ids = elec_dset['id'][:]

            std_cols = ['id', 'x', 'y', 'z', 'imp', 'location', 'filtering', 'group']
            extra_cols = [k for k in elec_dset.keys() if k not in std_cols and isinstance(elec_dset[k], h5py.Dataset)]

            for col in extra_cols:
                curr_colnames = nwbfile.electrodes.colnames if nwbfile.electrodes else ()
                if col not in curr_colnames:
                    desc = "N/A"
                    if 'description' in elec_dset[col].attrs:
                        desc = _safe_decode(elec_dset[col].attrs['description'])
                    nwbfile.add_electrode_column(name=col, description=desc)

            for i in range(len(ids)):
                eg_name = 'default'
                try: eg_name = f[elec_dset['group'][i]].name.split('/')[-1]
                except: pass
                eg = eg_map.get(eg_name, list(eg_map.values())[0] if eg_map else nwbfile.create_electrode_group('default', 'auto', 'unknown', list(device_map.values())[0]))

                base_kwargs = {
                    'id': ids[i],
                    'x': elec_dset['x'][i] if 'x' in elec_dset else np.nan,
                    'y': elec_dset['y'][i] if 'y' in elec_dset else np.nan,
                    'z': elec_dset['z'][i] if 'z' in elec_dset else np.nan,
                    'imp': elec_dset['imp'][i] if 'imp' in elec_dset else np.nan,
                    'location': _safe_decode(elec_dset['location'][i]) if 'location' in elec_dset else 'unknown',
                    'filtering': _safe_decode(elec_dset['filtering'][i]) if 'filtering' in elec_dset else 'unknown',
                    'group': eg
                }
                for col in extra_cols:
                    base_kwargs[col] = _safe_decode(elec_dset[col][i])
                nwbfile.add_electrode(**base_kwargs)
        else:
            dev = list(device_map.values())[0]
            eg = nwbfile.create_electrode_group('dummy', 'dummy', 'unknown', dev)
            nwbfile.add_electrode(id=0, x=0.0, y=0.0, z=0.0, imp=0.0, location='unknown', filtering='none', group=eg)

        # 3. Intervals
        if 'intervals' in f:
            print("Inspecting: /intervals")
            for key in f['intervals']:
                print(f"  - Found Interval: {key}")
                grp = f['intervals'][key]
                ti = nwbfile.create_time_intervals(name=key, description=f"Intervals for {key}")
                colnames = [k for k in grp.keys() if k not in ['id', 'start_time', 'stop_time'] and isinstance(grp[k], h5py.Dataset)]
                for col in colnames: ti.add_column(name=col, description=col)
                for i in range(len(grp['id'])):
                    row = {'start_time': grp['start_time'][i], 'stop_time': grp['stop_time'][i]}
                    for col in colnames: row[col] = _safe_decode(grp[col][i])
                    ti.add_row(**row)

        # 4. Units
        if 'units' in f:
            print("Inspecting: /units")
            grp = f['units']

            units_desc = "N/A"
            if 'description' in grp.attrs:
                units_desc = _safe_decode(grp.attrs['description'])

            if nwbfile.units is None:
                nwbfile.units = Units(name='units', description=units_desc)
            else:
                nwbfile.units.description = units_desc

            if 'id' in grp:
                ids = grp['id'][:]
                col_names = [k for k in grp.keys() if isinstance(grp[k], h5py.Dataset) and k != 'id' and not k.endswith('_index')]
                for col in col_names:
                    if col == 'spike_times': continue
                    curr_colnames = nwbfile.units.colnames if nwbfile.units else ()
                    if col not in curr_colnames:
                        desc = "N/A"
                        if 'description' in grp[col].attrs:
                             desc = _safe_decode(grp[col].attrs['description'])
                        nwbfile.add_unit_column(name=col, description=desc)

                st_data = grp['spike_times'][:] if 'spike_times' in grp else None
                st_index = grp['spike_times_index'][:] if 'spike_times_index' in grp else None

                for i, unit_id in enumerate(ids):
                    row_data = {'id': unit_id}
                    for col in col_names:
                        if col == 'spike_times': continue
                        row_data[col] = _safe_decode(grp[col][i])
                    if st_data is not None and st_index is not None:
                        start = st_index[i-1] if i > 0 else 0
                        end = st_index[i]
                        row_data['spike_times'] = st_data[start:end]
                    elif st_data is not None and st_index is None and len(ids) == 1:
                         row_data['spike_times'] = st_data
                    nwbfile.add_unit(**row_data)

        # 5. Acquisition
        if 'acquisition' in f:
            print("Inspecting: /acquisition")
            for key in f['acquisition']:
                print(f"  - Found Acquisition: {key}")
                grp = f['acquisition'][key]
                target_grp = None

                if key + '_data' in grp: target_grp = grp[key + '_data']
                elif 'data' in grp: target_grp = grp
                else:
                     for sub_key in grp:
                         sub_item = grp[sub_key]
                         if isinstance(sub_item, h5py.Group) and 'data' in sub_item and 'timestamps' in sub_item:
                             target_grp = sub_item
                             break

                if target_grp and 'data' in target_grp and 'timestamps' in target_grp:
                    dset = target_grp['data']
                    ts = target_grp['timestamps']

                    if 'lfp' in key.lower() or 'muae' in key.lower():
                        if 'electrodes' in target_grp: elec_idxs = target_grp['electrodes'][:]
                        else: elec_idxs = list(range(min(dset.shape[1], len(nwbfile.electrodes))))
                        elec_region = nwbfile.create_electrode_table_region(region=list(range(len(elec_idxs))), description=f"Electrodes for {key}")
                        es = ElectricalSeries(name=key, data=DataChunkIterator(dset, buffer_size=20000), timestamps=DataChunkIterator(ts, buffer_size=20000), electrodes=elec_region, description=f"Reconstructed {key}")
                        nwbfile.add_acquisition(es)
                    else:
                        ts_obj = TimeSeries(name=key, data=DataChunkIterator(dset, buffer_size=20000), timestamps=DataChunkIterator(ts, buffer_size=20000), unit='unknown', description=f"Reconstructed {key}")
                        nwbfile.add_acquisition(ts_obj)

        # 6. Processing
        if 'processing' in f:
            print("Inspecting: /processing")
            for mod_key in f['processing']:
                print(f"  - Found Module: {mod_key}")
                desc = f"Reconstructed {mod_key}"
                proc_mod = nwbfile.create_processing_module(name=mod_key, description=desc)
                mod_grp = f['processing'][mod_key]
                for sub_key in mod_grp:
                    sub_item = mod_grp[sub_key]
                    if isinstance(sub_item, h5py.Group) and 'data' in sub_item and 'timestamps' in sub_item:
                        dset = sub_item['data']
                        ts = sub_item['timestamps']
                        ts_obj = TimeSeries(name=sub_key, data=DataChunkIterator(dset, buffer_size=20000), timestamps=DataChunkIterator(ts, buffer_size=20000), unit='unknown', description=f"Reconstructed from {mod_key}/{sub_key}")
                        proc_mod.add(ts_obj)

                if mod_key == 'spike_train' and (nwbfile.units is None or len(nwbfile.units) == 0):
                     print(f"    -> Populating Units from {mod_key} (Alternative Source)")
                     try:
                         sub_key = mod_key + '_data'
                         if sub_key in mod_grp:
                             data_node = mod_grp[sub_key]['data']
                             ts_node = mod_grp[sub_key]['timestamps']
                             spike_data = data_node[:]
                             timestamps = ts_node[:]
                             elec_map = None
                             if 'electrodes' in mod_grp[sub_key]:
                                 elec_map = mod_grp[sub_key]['electrodes'][:]
                                 nwbfile.add_unit_column(name='electrode_id', description='Electrode ID from spike_train')
                             n_units = spike_data.shape[1]
                             for u in range(n_units):
                                 spikes = timestamps[np.nonzero(spike_data[:, u])[0]]
                                 extra_kwargs = {}
                                 if elec_map is not None and u < len(elec_map): extra_kwargs['electrode_id'] = int(elec_map[u])
                                 nwbfile.add_unit(spike_times=spikes, id=u, **extra_kwargs)
                     except Exception as e:
                         print(f"Error processing spike train units {mod_key}: {e}")

        # 7. Stimulus
        if 'stimulus' in f:
            print("Inspecting: /stimulus")
            if 'presentation' in f['stimulus']:
                print("  - Found presentation")
                stim_grp = f['stimulus']['presentation']
                for key in stim_grp:
                     if isinstance(stim_grp[key], h5py.Group) and 'data' in stim_grp[key] and 'timestamps' in stim_grp[key]:
                         dset = stim_grp[key]['data']
                         ts = stim_grp[key]['timestamps']
                         ts_obj = TimeSeries(name=key, data=DataChunkIterator(dset, buffer_size=20000), timestamps=DataChunkIterator(ts, buffer_size=20000), unit='unknown', description=f"Reconstructed stimulus {key}")
                         nwbfile.add_stimulus(ts_obj)

        # 8. Scratch
        if 'scratch' in f:
             print("Inspecting: /scratch")
             scratch_grp = f['scratch']
             for key in scratch_grp:
                 if isinstance(scratch_grp[key], h5py.Dataset):
                     print(f"  - Found scratch dataset: {key}")
                     nwbfile.add_scratch(scratch_grp[key][()], name=key, description="Reconstructed scratch")

        # Write
        print(f"Writing to {target_filepath}...")
        with NWBHDF5IO(target_filepath, 'w') as io:
            io.write(nwbfile)
        print("Reconstruction Complete.")


def get_binary_events_for_code(nwb_file, target_code=50.0, target_interval_name=None, code_column='codes'):
    """
    Extracts binary event indicators from an NWBFile object based on a target_code
    in a specified interval table.

    A '1' in the returned NumPy array indicates that the `code_column` in the specified
    `interval_table` matches the `target_code` at that row, and '0' otherwise.

    Args:
        nwb_file (pynwb.NWBFile): The NWBFile object to inspect.
        target_code (float): The code to match in the `code_column`.
        target_interval_name (str, optional): The name of a specific interval table to search.
                                            If None, the function will return an empty array
                                            if no specific table is designated.
        code_column (str): The name of the column in the interval table that contains the codes.
                           Defaults to 'codes'.

    Returns:
        numpy.ndarray: A binary NumPy array (1s and 0s) indicating rows where the code matches,
                       or an empty array if conditions are not met.
    """
    if not hasattr(nwb_file, 'intervals') or not nwb_file.intervals:
        print("No interval tables found in the NWB file.")
        return np.array([])

    if not target_interval_name or target_interval_name not in nwb_file.intervals:
        print(f"Warning: Specified interval table '{target_interval_name}' not found or not provided. Returning empty array.")
        return np.array([])

    interval_table = nwb_file.intervals[target_interval_name]
    df = interval_table.to_dataframe()

    if code_column not in df.columns:
        print(f"Warning: Interval table '{target_interval_name}' does not contain '{code_column}' column. Returning array of zeros.")
        return np.zeros(len(df), dtype=int)

    if 'start_time' not in df.columns:
        print(f"Warning: Interval table '{target_interval_name}' does not contain 'start_time' column. This may affect interpretation.")

    # Convert the column to numeric, coercing errors to NaN
    codes_for_comparison = pd.to_numeric(df[code_column], errors='coerce')

    # Create a boolean mask using robust floating-point comparison
    # np.isclose handles potential floating point inaccuracies. equal_nan=False treats NaNs as not equal.
    binary_mask = np.isclose(codes_for_comparison, target_code, equal_nan=False)

    # Convert boolean mask to integer array (True -> 1, False -> 0)
    binary_array = binary_mask.astype(int)

    return binary_array


def get_onset_time_bin(nwb_file, binary_event_array, target_interval_name):
    """
    Retrieves start_time values from the specified interval table based on a binary event array.

    Args:
        nwb_file (pynwb.NWBFile): The NWBFile object to inspect.
        binary_event_array (numpy.ndarray): A binary NumPy array where '1' indicates an event.
        target_interval_name (str): The name of the interval table from which to extract start_times.

    Returns:
        list: A list of start_time values corresponding to '1's in the binary_event_array.
              Returns an empty list if the interval table is not found or 'start_time' column is missing.
    """
    onset_times = []

    if not hasattr(nwb_file, 'intervals') or not nwb_file.intervals:
        print("No interval tables found in the NWB file.")
        return onset_times

    if not target_interval_name or target_interval_name not in nwb_file.intervals:
        print(f"Warning: Specified interval table '{target_interval_name}' not found or not provided. Returning empty list.")
        return onset_times

    interval_table = nwb_file.intervals[target_interval_name]
    df = interval_table.to_dataframe()

    if 'start_time' not in df.columns:
        print(f"Warning: Interval table '{target_interval_name}' does not contain 'start_time' column. Returning empty list.")
        return onset_times

    # Ensure binary_event_array matches the length of the DataFrame
    if len(binary_event_array) != len(df):
        print("Error: Length of binary_event_array does not match the length of the interval table. Returning empty list.")
        return onset_times

    # Filter start_times where the binary_event_array is 1
    onset_times = df['start_time'][binary_event_array == 1].tolist()

    return onset_times


def get_signal_array(nwb_file, event_timestamps, time_pre, time_post, signal_mode='lfp', probe_id=0, eye_dimension_index=0):
    signal_data_h5 = None
    signal_timestamps_h5 = None
    num_channels = 1
    signal_name = ""

    if signal_mode == 'lfp':
        signal_name = f'probe_{probe_id}_lfp'
        if signal_name in nwb_file.acquisition:
            electrical_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = electrical_series.data # Store h5py.Dataset reference
            signal_timestamps_h5 = electrical_series.timestamps # Store h5py.Dataset reference
            num_channels = signal_data_h5.shape[1] if signal_data_h5.ndim > 1 else 1
        else:
            print(f"Error: LFP data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'muae':
        signal_name = f'probe_{probe_id}_muae'
        if signal_name in nwb_file.acquisition:
            electrical_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = electrical_series.data # Store h5py.Dataset reference
            signal_timestamps_h5 = electrical_series.timestamps # Store h5py.Dataset reference
            num_channels = signal_data_h5.shape[1] if signal_data_h5.ndim > 1 else 1
        else:
            print(f"Error: MUAe data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'pupil':
        signal_name = 'pupil_1_tracking'
        if signal_name in nwb_file.acquisition:
            time_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = time_series.data # Store h5py.Dataset reference
            signal_timestamps_h5 = time_series.timestamps # Store h5py.Dataset reference
            num_channels = 1 # Pupil is usually 1D
        else:
            print(f"Error: Pupil tracking data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'eye':
        signal_name = 'eye_1_tracking'
        if signal_name in nwb_file.acquisition:
            time_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = time_series.data # Store h5py.Dataset reference
            signal_timestamps_h5 = time_series.timestamps # Store h5py.Dataset reference
            # Eye tracking data typically has 2 dimensions (X, Y)
            if signal_data_h5.ndim > 1 and signal_data_h5.shape[1] > eye_dimension_index:
                # We will slice this in the loop, so keep the full h5py.Dataset for now
                pass
            else:
                print(f"Warning: Eye tracking data has unexpected dimensions or eye_dimension_index {eye_dimension_index} is out of bounds. Using first dimension.")
            num_channels = 1 # We extract one dimension at a time for 'eye' mode
        else:
            print(f"Error: Eye tracking data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'photodiode': # Added photodiode handling
        signal_name = 'photodiode_1_tracking'
        if signal_name in nwb_file.acquisition:
            time_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = time_series.data # Store h5py.Dataset reference
            signal_timestamps_h5 = time_series.timestamps # Store h5py.Dataset reference
            num_channels = 1
        else:
            print(f"Error: Photodiode tracking data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'convolved_spike_train':
        signal_name = 'convolved_spike_train_data'
        if 'convolved_spike_train' in nwb_file.processing and signal_name in nwb_file.processing['convolved_spike_train'].data_interfaces:
            time_series = nwb_file.processing['convolved_spike_train'].data_interfaces[signal_name]
            signal_data_h5 = time_series.data # Store h5py.Dataset reference
            signal_timestamps_h5 = time_series.timestamps # Store h5py.Dataset reference
            num_channels = signal_data_h5.shape[1] if signal_data_h5.ndim > 1 else 1
        else:
            print(f"Error: Convolved spike train data for {signal_name} not found.")
            return np.array([])
    else:
        print(f"Error: Invalid signal_mode '{signal_mode}'. Supported modes are 'lfp', 'muae', 'pupil', 'eye', 'photodiode', 'convolved_spike_train'.")
        return np.array([])

    if signal_data_h5 is None or signal_timestamps_h5 is None or len(signal_timestamps_h5) == 0:
        print(f"Error: No data or timestamps found for signal_mode '{signal_mode}'.")
        return np.array([])

    # Load timestamps fully, as they are typically smaller and needed for searchsorted/diff
    signal_timestamps = signal_timestamps_h5[:]

    if len(signal_timestamps) > 1:
        sampling_rate = 1 / np.mean(np.diff(signal_timestamps))
    else:
        print("Warning: Not enough timestamps to calculate sampling rate, assuming 1000 Hz.")
        sampling_rate = 1000.0

    num_time_points_in_window = int(np.round((time_pre + time_post) * sampling_rate))

    # Use dtype from the h5py.Dataset object
    if num_channels > 1 and signal_mode not in ['pupil', 'eye', 'photodiode']:
        result_array = np.full((len(event_timestamps), num_time_points_in_window, num_channels), np.nan, dtype=signal_data_h5.dtype)
    else:
        result_array = np.full((len(event_timestamps), num_time_points_in_window), np.nan, dtype=signal_data_h5.dtype)

    for i, event_ts in enumerate(event_timestamps):
        window_start_time = event_ts - time_pre
        window_end_time = event_ts + time_post

        start_idx = np.searchsorted(signal_timestamps, window_start_time, side='left')
        end_idx = np.searchsorted(signal_timestamps, window_end_time, side='right')

        data_segment_start_idx = max(0, start_idx)
        data_segment_end_idx = min(len(signal_timestamps_h5), end_idx)

        actual_samples_from_signal = data_segment_end_idx - data_segment_start_idx

        if actual_samples_from_signal <= 0:
            continue

        # Calculate the ideal start and end indices within the result_array for this event
        ideal_paste_start = int(np.round((signal_timestamps[data_segment_start_idx] - window_start_time) * sampling_rate))
        ideal_paste_end = ideal_paste_start + actual_samples_from_signal

        # Clamp these indices to the actual bounds of the result_array's window
        final_result_slice_start = max(0, ideal_paste_start)
        final_result_slice_end = min(num_time_points_in_window, ideal_paste_end)

        # Determine the corresponding slice within the current_data_segment
        source_data_start_offset = final_result_slice_start - ideal_paste_start
        source_data_end_offset = source_data_start_offset + (final_result_slice_end - final_result_slice_start)

        # Extract the relevant data segment by slicing the h5py.Dataset object
        if signal_mode == 'eye' and signal_data_h5.ndim > 1:
            current_data_segment = signal_data_h5[data_segment_start_idx:data_segment_end_idx, eye_dimension_index]
        else:
            current_data_segment = signal_data_h5[data_segment_start_idx:data_segment_end_idx]

        # Take the correct part of the extracted segment that fits into the result array
        segment_to_copy = current_data_segment[source_data_start_offset : source_data_end_offset]

        # Place data into the result array
        if num_channels > 1 and signal_mode not in ['pupil', 'eye', 'photodiode']:
            result_array[i, final_result_slice_start:final_result_slice_end, :] = segment_to_copy
        else:
            result_array[i, final_result_slice_start:final_result_slice_end] = segment_to_copy

    return result_array


def get_unit_column_data(nwb_file, column_label):
    """
    Retrieves data from a specified column of the nwb.units table.

    Args:
        nwb_file (pynwb.NWBFile): The NWBFile object containing the units data.
        column_label (str): The name of the column to retrieve (e.g., "snr", "presence_ratio").

    Returns:
        pandas.Series or None: A pandas Series containing the data from the specified column,
                              or None if the units table or column does not exist.
    """
    if nwb_file.units is None or len(nwb_file.units) == 0:
        print("Units table is empty or does not exist in the NWB file.")
        return None

    if column_label not in nwb_file.units.colnames:
        print(f"Column '{column_label}' not found in the units table. Available columns: {nwb_file.units.colnames}")
        return None

    df_units = nwb_file.units.to_dataframe()
    return df_units[column_label]


def get_neuron_info(nwb, unit_id):
    """
    Retrieves info for a specific neuron by ID.

    Args:
        nwb: The NWBFile object.
        unit_id: The ID of the neuron unit.

    Returns:
        peak_channel, id, snr, presence_ratio, area
    """
    if nwb.units is None:
        print("No units table found.")
        return None, unit_id, None, None, None

    # Get all unit IDs to find the index of the requested unit_id
    all_ids = nwb.units.id[:]

    try:
        index = list(all_ids).index(unit_id)
    except ValueError:
        print(f"Unit ID {unit_id} not found in nwb.units.")
        return None, unit_id, None, None, None

    # Helper to safely retrieve column data
    def get_col_val(col_name, idx):
        if col_name in nwb.units.colnames:
            return nwb.units[col_name][idx]
        return float('nan')

    # Retrieve requested values
    peak_channel = get_col_val('peak_channel_id', index)
    snr = get_col_val('snr', index)
    presence_ratio = get_col_val('presence_ratio', index)

    # Find Area from electrodes table
    area = "unknown"
    if nwb.electrodes is not None:
        try:
            # Convert peak_channel to int ID (handle string '3.0' -> 3)
            elec_id = int(float(peak_channel))

            # Find index in electrodes table
            # nwb.electrodes.id is a dataset containing IDs
            elec_ids = nwb.electrodes.id[:]
            if elec_id in elec_ids:
                elec_idx = list(elec_ids).index(ele_id)

                # Try 'location' first, usually holds area info
                if 'location' in nwb.electrodes.colnames:
                    val = nwb.electrodes['location'][elec_idx]
                    # Handle bytes vs string
                    area = val.decode('utf-8') if isinstance(val, bytes) else str(val)
                elif 'label' in nwb.electrodes.colnames:
                    val = nwb.electrodes['label'][elec_idx]
                    area = val.decode('utf-8') if isinstance(val, bytes) else str(val)
        except Exception as e:
            # area remains "unknown" or could be set to error message
            pass

    return peak_channel, unit_id, snr, presence_ratio, area


def get_unit_ids_for_area(nwb_file, target_area_name):
    """
    Retrieves the IDs of units located in a specified brain area from an NWBFile.

    Args:
        nwb_file (pynwb.NWBFile): The NWBFile object containing the units and electrodes data.
        target_area_name (str): The name of the brain area to search for (e.g., "PFC", "V4, MT").

    Returns:
        list: A list of unit IDs that belong to the specified area.
              Returns an empty list if no units are found for the area or if data is missing.
    """
    if nwb_file.units is None or len(nwb_file.units) == 0:
        print("No units table found or it is empty in the NWB file.")
        return []

    if nwb_file.electrodes is None or len(nwb_file.electrodes) == 0:
        print("No electrodes table found or it is empty in the NWB file.")
        return []

    units_df = nwb_file.units.to_dataframe().reset_index() # Make 'id' a column from the index
    # Reset index of electrodes_df to ensure 'id' is a column, not just an index name
    electrodes_df = nwb_file.electrodes.to_dataframe().reset_index()

    # Check for essential columns after ensuring 'id' is a column
    if 'peak_channel_id' not in units_df.columns:
        print("Units table is missing 'peak_channel_id' column.")
        return []
    if 'id' not in electrodes_df.columns:
        print("Error: 'id' column not found in electrodes DataFrame after resetting index. This should not happen.")
        return []

    # Determine the correct column for area information in electrodes_df
    area_col_name = None
    if 'location' in electrodes_df.columns:
        area_col_name = 'location'
    elif 'label' in electrodes_df.columns:
        area_col_name = 'label'
    else:
        print("Electrodes table is missing 'location' or 'label' column for area information.")
        return []

    # Merge units_df with electrodes_df to get area information for each unit
    # Convert peak_channel_id to int to match electrode_id type
    # Handle cases where peak_channel_id might be float-like strings (e.g., '3.0')
    units_df['peak_channel_id'] = units_df['peak_channel_id'].astype(float).astype(int)

    # Ensure electrode 'id' column is also integer type for consistent merge
    electrodes_df['id'] = electrodes_df['id'].astype(int)

    merged_df = pd.merge(
        units_df,
        electrodes_df[[area_col_name, 'id']],
        left_on='peak_channel_id',
        right_on='id',
        how='left',
        suffixes=('_unit', '_electrode')
    )

    # Normalize area names for comparison (handle bytes and case insensitivity)
    merged_df['area_normalized'] = merged_df[area_col_name].apply(lambda x: x.decode('utf-8').strip().upper() if isinstance(x, bytes) else str(x).strip().upper())
    target_area_name_normalized = target_area_name.strip().upper()

    # Filter units by the target area name
    filtered_units = merged_df[merged_df['area_normalized'] == target_area_name_normalized]

    # Return the unit IDs as a list
    return filtered_units['id_unit'].tolist()


class vFLIP2:
    """
    vFLIP2 Analysis Class

    Analyzes electrophysiological data to identify spectrolaminar motifs based on
    power changes across frequency bands and laminar depth.
    """

    def __init__(self, data,
                 intdist=np.nan,
                 freqbinsize=1.0,
                 DataType='psd',
                 fsample=np.nan,
                 orientation='both',
                 layer4Thickness=np.nan,
                 plot_result=False,
                 omega_cut=6.0):

        # Input Validation
        if DataType not in ["psd", "raw", "raw_cut"]:
            raise ValueError("DataType must be 'psd', 'raw', or 'raw_cut'.")
        if orientation not in ["upright", "inverted", "both"]:
            raise ValueError("orientation must be 'upright', 'inverted', or 'both'.")

        self.plot_combined = False
        self.omega_cut = omega_cut

        # Handle inter-channel distance
        if np.isnan(intdist):
            try:
                val = float(input('Please enter the interchannel distance in mm (intdist): '))
                if val <= 0: raise ValueError
                self.intdist = val
            except:
                raise ValueError('Invalid interchannel distance entered.')
        else:
            self.intdist = intdist

        # Setup step sizes
        self.step = int(round(0.1 / self.intdist))  # search steps on channels
        self.minrange_s = int(np.ceil(0.3 / self.intdist))

        # Handle Data Input
        if DataType == "psd":
            self.nonnormpowmat = data
            self.freqbinsize = freqbinsize
        elif DataType in ["raw", "raw_cut"]:
            if np.isnan(fsample):
                try:
                    val = float(input('Please enter the sampling rate (fsample): '))
                    if val <= 0: raise ValueError
                    self.fsample = val
                except:
                    raise ValueError('Invalid sampling rate entered.')
            else:
                self.fsample = fsample

            if DataType == "raw":
                # Assuming data is (n_chan, n_time)
                trials = self._split_into_trials(data)
                self.nonnormpowmat = self._compute_psd_hanning(trials)
            elif DataType == "raw_cut":
                # Assuming data is list of arrays or 3D array
                self.nonnormpowmat = self._compute_psd_hanning(data)

            self.freqbinsize = 1.0

        # Handle NaNs and Row trimming
        # Check first column for NaNs to determine valid channel range
        nanboolean = ~np.isnan(self.nonnormpowmat[:, 0])
        if np.sum(nanboolean) == 0:
            raise ValueError("Error using FLIPAnalysis: Empty matrix")

        # Find first and last valid row indices
        valid_indices = np.where(nanboolean)[0]
        self.startrow = valid_indices[0]
        self.endrow = valid_indices[-1] # Python index inclusive for logic, careful with slicing

        self.freqaxis = np.arange(1, self.nonnormpowmat.shape[1] + 1) * self.freqbinsize

        # Orientation setup
        if orientation == 'both':
            self.orientation1 = 0
        elif orientation == 'upright':
            self.orientation1 = 1
        elif orientation == "inverted":
            self.orientation1 = -1

        # Layer 4 Thickness
        # Citation: O'Kusky, J., & Colonnier, M. (1982).
        laminae_thickness_mm = np.array([122.9, 396.9, 127.1, 211.4, 247.5, 226.3, 260.2]) / 1000.0

        if np.isnan(layer4Thickness):
            # Sum of indices 2, 3, 4 (IVA, IVB, IVC) -> Python indices 2:5
            layer4 = np.sum(laminae_thickness_mm[2:5])
        else:
            layer4 = layer4Thickness

        self.minrange = int(np.ceil(layer4 / self.intdist))

        # Run Analysis
        self.Results = self.flip_it()
        self.relpow = self._get_Window(self.startrow, self.endrow)

        if self.Results is not None and plot_result:
            self.plot_result()

    # =========================================================================
    # CORE ANALYSIS METHODS
    # =========================================================================

    def _get_Window(self, proximalchannel, distalchannel):
        # Slicing: inclusive of proximal, inclusive of distal (so +1 for Python)
        powspec_window = self.nonnormpowmat[proximalchannel : distalchannel + 1, :]

        # Max power along freq axis (dim 0 in subarray, which corresponds to channels)
        # MATLAB: max(A, [], 1) returns row vector of maxes of each column.
        # Wait, MATLAB: max(powspec_window,[],1) finds max across DIM 1 (channels).
        # Result is (1, n_freqs).
        # The code calculates relative power by dividing by max power across the depth for that frequency.

        maxpow = np.max(powspec_window, axis=0)

        # Avoid divide by zero
        maxpow[maxpow == 0] = np.nan

        relpow = powspec_window / maxpow
        return relpow

    def _get_freqbands(self, S1_meanpow, S2_meanpow):
        """
        Determines deep and superficial frequency bands.
        """
        def find_longest_true_run(logical_array):
            # Pad with 0 to detect edges
            d = np.diff(np.concatenate(([0], logical_array.astype(int), [0])))
            run_starts = np.where(d == 1)[0]
            run_ends = np.where(d == -1)[0] - 1
            run_lengths = run_ends - run_starts + 1

            if len(run_lengths) > 0:
                idx = np.argmax(run_lengths)
                # Return range indices
                return np.arange(run_starts[idx], run_ends[idx] + 1)
            else:
                return np.array([], dtype=int)

        # Boolean masks for frequency ranges
        lowfreqs = (self.freqaxis > 4) & (self.freqaxis < 70)
        highfreqs = (self.freqaxis > 40) & (self.freqaxis < 150)

        # Comparisons
        greater = (S1_meanpow > S2_meanpow)
        lesser = (S1_meanpow < S2_meanpow)

        longest_run_P1 = find_longest_true_run(greater & lowfreqs)
        longest_run_P2 = find_longest_true_run(lesser & lowfreqs)

        len_P1 = len(longest_run_P1)
        len_P2 = len(longest_run_P2)

        deep_f = []
        sup_f = []
        orientation = 0
        ind_high = None

        # Determine dominant low frequency group
        if len_P1 >= 5 and len_P2 >= 5:
            freq_range_P1 = self.freqaxis[longest_run_P1]
            freq_range_P2 = self.freqaxis[longest_run_P2]

            if np.min(freq_range_P1) < np.min(freq_range_P2):
                deep_f = freq_range_P1
                orientation = -1
                ind_high = lesser & highfreqs
            else:
                deep_f = freq_range_P2
                orientation = 1
                ind_high = greater & highfreqs
        elif len_P1 >= 5:
            deep_f = self.freqaxis[longest_run_P1]
            orientation = -1
            ind_high = lesser & highfreqs
        elif len_P2 >= 5:
            deep_f = self.freqaxis[longest_run_P2]
            orientation = 1
            ind_high = greater & highfreqs

        # Determine superficial frequency based on orientation
        if orientation != 0 and ind_high is not None:
            # Need indices of freqaxis where ind_high is true
            longest_run_high = find_longest_true_run(ind_high)
            if len(longest_run_high) > 0:
                sup_f = self.freqaxis[longest_run_high]
                if len(sup_f) < 20 or sup_f[-1] < 70:
                    sup_f = []
            else:
                sup_f = []

        # Convert empty numpy arrays to empty lists or None for consistency if needed,
        # but maintaining numpy array is usually better for indexing later.
        if len(deep_f) == 0: deep_f = []
        if len(sup_f) == 0: sup_f = []

        return deep_f, sup_f, orientation

    def _peak_check(self, band, proximalchannel, distalchannel):
        # Find indices where band is max
        peak_locations = np.where(band == np.max(band))[0]

        if len(peak_locations) == 0: return False

        if np.mean(peak_locations) > len(band) / 2:
            peak_index = np.max(peak_locations)
        else:
            peak_index = np.min(peak_locations)

        # Check boundary conditions and local maxima
        # Note: Python indices 0 to len-1
        if peak_index == 0:
            # proximalchannel == self.startrow check
            is_edge = (proximalchannel == self.startrow)
            is_decreasing = (band[peak_index] > band[peak_index + 1])
            peak_max_check = is_edge and is_decreasing
        elif peak_index == len(band) - 1:
            is_edge = (distalchannel == self.endrow)
            is_increasing = (band[peak_index] > band[peak_index - 1])
            peak_max_check = is_edge and is_increasing
        else:
            peak_max_check = (band[peak_index] > band[peak_index + 1]) and \
                             (band[peak_index] > band[peak_index - 1])

        n = len(band)
        # MATLAB: idx <= minrange_s OR idx >= n - (minrange_s - 1)
        # Note: MATLAB 1-based, Python 0-based
        # idx (matlab) = peak_index + 1
        # check1: (peak_index + 1 <= minrange_s) ...

        # Simplified logic for Python indices:
        # Check if peak is within the buffer zones at edges
        # minrange_s is a scalar count

        check1 = (peak_index < self.minrange_s) or (peak_index >= n - self.minrange_s)

        return peak_max_check and check1

    def _crossover_channels(self, lowband, highband, proximalchannel, orientation):
        band_diff = np.abs(highband - lowband)
        n = len(lowband)

        crossoverchannels = []

        # Inner helper logic
        def determine_cross(idx):
            # idx corresponds to Python index 0...n-3
            # Accessing idx, idx+1, idx+2

            b1 = highband if orientation > 0 else lowband
            b2 = lowband if orientation > 0 else highband

            # Condition 1: Direct Cross
            # b1[i] > b2[i] AND b2[i+1] > b1[i+1]
            if b1[idx] > b2[idx] and b2[idx+1] > b1[idx+1]:
                if abs(band_diff[idx]) <= abs(band_diff[idx+1]):
                    return idx
                else:
                    return idx + 1

            # Condition 2: Plateau Cross
            # b1[i] > b2[i] AND b1[i+1] == b2[i+1] AND b2[i+2] > b1[i+2]
            # Ensure idx+2 exists
            if idx + 2 < len(b1):
                if b1[idx] > b2[idx] and b1[idx+1] == b2[idx+1] and b2[idx+2] > b1[idx+2]:
                    return idx + 1

            return np.nan

        # Loop through channels (up to n-2 to allow for i+1 check, logic handles i+2 internally)
        for i in range(n - 1):
            res = determine_cross(i)
            if not np.isnan(res):
                crossoverchannels.append(res)

        crossoverchannels = np.array(crossoverchannels)

        if len(crossoverchannels) == 0:
            return np.nan
        elif len(crossoverchannels) == 1:
            # Convert relative channel index to absolute channel index
            return crossoverchannels[0] + proximalchannel # +1? No, indices align.
        else:
            # Multiple crosses, find the best one based on area difference
            ratings = []
            for cross_idx in crossoverchannels:
                cross_idx_int = int(cross_idx)
                # Sum differences before and after
                # Python slicing: 0:cross_idx includes up to cross_idx-1
                # MATLAB: 1:crossover_choice

                # Careful: The crossover index returned represents a specific channel.
                # The area calculation should split at that channel.

                # MATLAB: sum(diff(1:cross)) - sum(diff(cross:end))
                diff_sum_pre = np.sum(band_diff[:cross_idx_int+1])
                diff_sum_post = np.sum(band_diff[cross_idx_int:])

                ratings.append(diff_sum_pre - diff_sum_post)

            best_idx = np.argmax(ratings)
            return crossoverchannels[best_idx] + proximalchannel

    def _evaluate_individual_goodness(self, lowband, highband):
        set_pval = 0.05

        def BandRegress(band):
            n = len(band)
            x = np.arange(1, n + 1)
            # Polyfit degree 2: p[0]x^2 + p[1]x + p[2]
            p, residuals, _, _, _ = np.polyfit(x, band, 2, full=True)

            # Calculate R-squared
            y_pred = np.polyval(p, x)
            sst = np.sum((band - np.mean(band))**2)
            sse = np.sum((band - y_pred)**2)
            rsquared = 1 - (sse / sst) if sst != 0 else 0

            # Calculate p-value (approximate for polyfit)
            # Using F-statistic approach for model significance
            dof_model = 2
            dof_resid = n - (dof_model + 1)
            if dof_resid > 0 and sse > 0:
                msr = (sst - sse) / dof_model
                mse = sse / dof_resid
                f_stat = msr / mse
                # To avoid strict dependency on scipy.stats if not strictly needed,
                # but we imported scipy. let's use it implicitly or simplify.
                # Implementing simple check or assuming scipy available:
                pval = 1 - scipy.stats.f.cdf(f_stat, dof_model, dof_resid)
            else:
                pval = 1.0 # Not significant

            midpoint = np.round(n / 2)
            slope = 2 * p[0] * midpoint + p[1] # Derivative at midpoint
            return slope, rsquared, pval

        low_slope, low_r2, low_pval = BandRegress(lowband)
        high_slope, high_r2, high_pval = BandRegress(highband)

        goodness = high_r2 * low_r2
        significant = (low_pval < set_pval) and (high_pval < set_pval)

        if low_slope > 0 and high_slope < 0:
            Gsign = 1
        elif low_slope < 0 and high_slope > 0:
            Gsign = -1
        else:
            Gsign = 0

        return goodness * significant * Gsign

    def omega_fun(self):
        euc_distance = lambda g1, g2: np.sqrt(np.sum((g1 - g2)**2))

        best_split = np.full(12, np.nan)
        best_omega = -np.inf

        # Iterate Proximal
        # MATLAB: startrow : step : (endrow - minrange + 1)
        # Python range excludes stop, so we add logic
        prox_start = self.startrow
        prox_end = self.endrow - self.minrange + 1

        for proximalchannel in range(prox_start, prox_end + 1, self.step):

            # Iterate Distal
            dist_start = proximalchannel + self.minrange
            dist_end = self.endrow

            for distalchannel in range(dist_start, dist_end + 1, self.step):

                psd_normalized = self._get_Window(proximalchannel, distalchannel)

                # Determine local minrange_s
                self.minrange_s = int(np.floor(abs(proximalchannel - distalchannel) / 2))
                if self.minrange_s < 1: continue

                # Groups for freq band determination
                group1 = psd_normalized[:self.minrange_s, :]
                group2 = psd_normalized[-(self.minrange_s):, :] # Check indexing logic

                # Moving mean smoothing (window 5)
                # MATLAB smoothdata default is moving average.
                S1_meanpow = np.mean(group1, axis=0)
                S2_meanpow = np.mean(group2, axis=0)

                S1_meanpow = ndimage.uniform_filter1d(S1_meanpow, size=5)
                S2_meanpow = ndimage.uniform_filter1d(S2_meanpow, size=5)

                Ps_dist = euc_distance(S1_meanpow, S2_meanpow)

                deep_f, sup_f, orientation = self._get_freqbands(S1_meanpow, S2_meanpow)

                if len(deep_f) == 0 or len(sup_f) == 0:
                    continue

                # Indices for deep and sup freqs.
                # self.freqaxis contains the actual Hz. We need indices.
                # deep_f contains Hz values.
                # Assuming freqaxis matches indices 1-to-1 shifted by binsize if simple,
                # but better to find indices using searchsorted or isin if exact match.
                # Given construction: freqaxis = 1:size * binsize.

                # Logical masking to get bands from psd_normalized
                deep_mask = np.isin(self.freqaxis, deep_f)
                sup_mask = np.isin(self.freqaxis, sup_f)

                lowband = np.mean(psd_normalized[:, deep_mask], axis=1)
                highband = np.mean(psd_normalized[:, sup_mask], axis=1)

                band_dist = euc_distance(lowband, highband)
                goodness = self._evaluate_individual_goodness(lowband, highband)

                # Orientation check logic
                if self.orientation1 == -1 and goodness > 0:
                    goodness = 0
                elif self.orientation1 == 1 and goodness < 0:
                    goodness = 0

                omega = np.log(Ps_dist * band_dist * abs(goodness) * \
                               abs(proximalchannel - distalchannel) * \
                               len(deep_f) * len(sup_f))

                # Markers relative to the window
                # find(..., 1) returns first index
                high_max_idx = np.argmax(highband)
                low_max_idx = np.argmax(lowband)

                # Convert relative to absolute
                highfreqmaxchannel = high_max_idx + proximalchannel
                lowfreqmaxchannel = low_max_idx + proximalchannel

                crossover_point = self._crossover_channels(lowband, highband, proximalchannel, orientation)

                # Conditions
                adequate_difference = (omega != 0) and not np.isinf(omega)
                check_lowpeak = self._peak_check(lowband, proximalchannel, distalchannel)
                check_highpeak = self._peak_check(highband, proximalchannel, distalchannel)
                check_peak_dist = abs(highfreqmaxchannel - lowfreqmaxchannel) >= self.minrange
                valid_crossover = not np.isnan(crossover_point)

                good_arrangement = (
                    (lowfreqmaxchannel < crossover_point < highfreqmaxchannel) or
                    (lowfreqmaxchannel > crossover_point > highfreqmaxchannel)
                )

                non_overlap = (lowfreqmaxchannel != crossover_point) and \
                              (crossover_point != highfreqmaxchannel) and \
                              (lowfreqmaxchannel != highfreqmaxchannel)

                good_fit = adequate_difference and check_lowpeak and check_highpeak and \
                           valid_crossover and good_arrangement and non_overlap and check_peak_dist

                if good_fit and (omega > best_omega):
                    best_split = [
                        goodness,
                        deep_f[0], deep_f[-1],
                        sup_f[0], sup_f[-1],
                        proximalchannel, distalchannel,
                        lowfreqmaxchannel, highfreqmaxchannel,
                        crossover_point, omega, orientation
                    ]
                    best_omega = omega

        # Frequencies are already in Hz from get_freqbands logic,
        # but the MATLAB code multiplies indices by binsize at the end.
        # My _get_freqbands returns actual freq values, so no multiplication needed here.
        # However, to match MATLAB output structure exactly:
        return best_omega, best_split

    def flip_it(self):
        best_omega, best_split = self.omega_fun()

        if best_omega <= self.omega_cut:
            return None
        else:
            fields = ['goodnessvalue', 'startinglowfreq', 'endinglowfreq',
                      'startinghighfreq', 'endinghighfreq', 'proximalchannel',
                      'distalchannel', 'lowfreqmaxchannel', 'highfreqmaxchannel',
                      'crossoverchannel', 'omega', 'orientation']

            results = {}
            for i, field in enumerate(fields):
                results[field] = best_split[i]

            # Map struct-like object
            class ResultsStruct:
                def __init__(self, **entries):
                    self.__dict__.update(entries)
                    self.relpow = None

            return ResultsStruct(**results)

    # =========================================================================
    # SIGNAL PROCESSING
    # =========================================================================

    def _split_into_trials(self, data):
        # data: (n_channels, n_timepoints)
        trial_duration_sec = 1
        samples_per_trial = int(trial_duration_sec * self.fsample)
        n_channels, total_timepoints = data.shape
        num_trials = total_timepoints // samples_per_trial

        # Crop data to full trials
        cutoff = num_trials * samples_per_trial
        data_crop = data[:, :cutoff]

        # Reshape: (n_channels, samples_per_trial, num_trials)
        # MATLAB reshape fills columns first.
        # We want to split time axis.
        # Reshape to (n_channels, num_trials, samples_per_trial) then transpose?
        # Actually easier to use np.split or reshape logic.

        # Make (n_channels, num_trials, samples)
        reshaped = data_crop.reshape(n_channels, num_trials, samples_per_trial)

        # Return as list of (n_channels, samples) arrays to mimic cell array
        trials = [reshaped[:, i, :] for i in range(num_trials)]
        return trials

    def _compute_psd_hanning(self, data):
        # Data is list of (n_chan, n_sample) arrays or a 3D array (n_trial, n_chan, n_sample)
        # Note: MATLAB code input signature for raw_cut is (n, m, p) -> (chan, time, trial)

        is_list = isinstance(data, list)

        if is_list:
            ntrials = len(data)
            nchan = data[0].shape[0]
            ndatsamples = [d.shape[1] for d in data]
            max_ndatsample = max(ndatsamples)
        else:
            # Assumes (n_chan, n_time, n_trial) based on MATLAB reshape logic
            # But usually Python standard is (n_trial, n_chan, n_time).
            # Let's align with MATLAB reshape output: (nchan, samples, trials)
            # Actually my _split_into_trials outputs list.
            # If "raw_cut" is passed as array, assume (nchan, time, trials)
            nchan, max_ndatsample, ntrials = data.shape

        # Padding
        padding_len = int(2**np.ceil(np.log2(max_ndatsample)))
        pad_factor = padding_len / self.fsample
        endnsample = int(round(pad_factor * self.fsample))

        # Freqs of interest (1 to 150)
        foi = np.arange(1, 151)
        fboi = np.round(foi * pad_factor).astype(int) + 1 # MATLAB 1-based index logic
        # Python FFT freq indices: 0 is DC.
        # fboi in MATLAB maps to FFT bins.

        # Re-calc standard FFT bins
        freqs = np.fft.rfftfreq(endnsample, d=1/self.fsample)

        # Find indices closest to FOI
        # MATLAB logic creates specific indices. Let's replicate logic precisely.
        # MATLAB: fboi = round(foi * pad) + 1;
        # In Python index = round(foi * pad).
        # Because FFT[0] is 0Hz. FFT[1] is 1/(pad*fs) approx.
        freq_indices = np.round(foi * pad_factor).astype(int)

        # Setup output
        powspctrm = np.zeros((ntrials, nchan, len(freq_indices)))

        for itrial in range(ntrials):
            if is_list:
                dat = data[itrial]
            else:
                dat = data[:, :, itrial]

            n_chan_trial, ndatsample = dat.shape

            # Detrend (MATLAB code: dat - beta*x where x is ones -> subtracting mean)
            dat = dat - np.mean(dat, axis=1, keepdims=True)

            # Hanning Taper
            tap = np.hanning(ndatsample)
            tap = tap / np.linalg.norm(tap)

            # Apply Taper
            data_tap = dat * tap[np.newaxis, :]

            # Zero Pad
            postpad = endnsample - ndatsample
            if postpad > 0:
                data_tap_pad = np.pad(data_tap, ((0,0), (0, postpad)), 'constant')
            else:
                data_tap_pad = data_tap

            # FFT
            # MATLAB: fft(X, [], 2). Python: np.fft.fft(X, axis=1)
            dum = np.fft.fft(data_tap_pad, axis=1)

            # Extract Frequencies
            # MATLAB indices fboi. Python indices fboi-1 (since we did +1 in comment before)
            # Actually freq_indices calculated above should be correct for 0-based.
            dum = dum[:, freq_indices]

            # Phase correction (timedelay) - omitted for brevity as timedelay=time(1)
            # and usually time starts at 0 for trials.
            # In MATLAB code: time = time_step:time_step:ntrialtime.
            # So timedelay = 1/fsample.
            # If strictly replicating:
            timedelay = 1.0 / self.fsample
            if timedelay != 0:
                angletransform = np.zeros(len(freq_indices), dtype=complex)
                missedsamples = round(timedelay * self.fsample) # should be 1
                anglein = missedsamples * (2 * np.pi / self.fsample) * foi
                # e^(-i * angle)
                # MATLAB: atan2 logic.
                # Simplification: Apply phase shift
                phase_shift = np.exp(-1j * anglein)
                dum = dum * phase_shift[np.newaxis, :]

            # Scale
            dum = dum * np.sqrt(2 / endnsample)

            # Power
            powspctrm[itrial, :, :] = np.abs(dum)**2

        nonnormpow = powspctrm
        meanpow = np.mean(nonnormpow, axis=0) # Average over trials
        return meanpow

    # =========================================================================
    # PLOTTING
    # =========================================================================

    def plot_relpowMap(self, ax, plot_SLonly=False):
        if self.Results is None:
            print("No results to plot.")
            return

        if not plot_SLonly:
            start_idx = self.Results.proximalchannel
            end_idx = self.Results.distalchannel
            # Slice inclusive
            window_data = self.nonnormpowmat[start_idx : end_idx + 1, :]
            max_sp = np.max(window_data, axis=0)
            max_sp[max_sp == 0] = np.nan
            relpow1 = self.nonnormpowmat / max_sp

            s_chan = self.startrow
            e_chan = self.endrow

            # Crop relpow1 to valid analysis range for display
            # MATLAB: imagesc(relpow1) where relpow1 is the WHOLE matrix normalized by the window max?
            # Re-reading MATLAB:
            # relpow1 = squeeze(obj.nonnormpowmat)./max_sp;
            # It normalizes the ENTIRE matrix by the max power found in the PROX-DIST window.
            img_data = relpow1[s_chan : e_chan + 1, :]

        else:
            s_chan = self.Results.proximalchannel
            e_chan = self.Results.distalchannel
            relpow1 = self._get_Window(s_chan, e_chan)
            img_data = relpow1

        self.Results.relpow = img_data # Store for consistency

        im = ax.imshow(img_data, aspect='auto', cmap='jet',
                       extent=[self.freqaxis[0], self.freqaxis[-1], e_chan, s_chan])
        # Note: extent Y is top, bottom. standard imagesc puts low index at top.

        ax.set_title('LFP Relative Power')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Channel Number')

        # Colorbar
        plt.colorbar(im, ax=ax, label='Relative Power')
        im.set_clim(0.3, 1.0) # Matches setcb logic

        self._plot_freqran(ax)
        self._plot_physMarkers(ax, self.freqaxis[-1], plot_SLonly)

    def plot_bandedrelpow(self, ax, plot_SLonly=False):
        if self.Results is None: return

        if not plot_SLonly:
            start_idx = self.Results.proximalchannel
            end_idx = self.Results.distalchannel
            window_data = self.nonnormpowmat[start_idx : end_idx + 1, :]
            max_sp = np.max(window_data, axis=0)
            max_sp[max_sp == 0] = np.nan
            relpow1 = self.nonnormpowmat / max_sp

            s_chan = self.startrow
            e_chan = self.endrow
            plot_data = relpow1[s_chan : e_chan + 1, :]
        else:
            s_chan = self.Results.proximalchannel
            e_chan = self.Results.distalchannel
            plot_data = self._get_Window(s_chan, e_chan)

        # Get indices for bands
        # self.Results.startinglowfreq is in Hz. Convert to indices.
        # Assuming freqbinsize = 1, indices roughly match Hz-1.
        # Using searchsorted for robustness.
        l_start = np.searchsorted(self.freqaxis, self.Results.startinglowfreq)
        l_end = np.searchsorted(self.freqaxis, self.Results.endinglowfreq)
        h_start = np.searchsorted(self.freqaxis, self.Results.startinghighfreq)
        h_end = np.searchsorted(self.freqaxis, self.Results.endinghighfreq)

        lowband = np.mean(plot_data[:, l_start : l_end + 1], axis=1)
        highband = np.mean(plot_data[:, h_start : h_end + 1], axis=1)

        channels = np.arange(s_chan, e_chan + 1)

        ax.plot(lowband, channels, 'b', linewidth=2, label='Low Band')
        ax.plot(highband, channels, 'r', linewidth=2, label='High Band')
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel('Relative Power')
        ax.set_ylabel('Channel Number')

        self._plot_physMarkers(ax, 1, plot_SLonly)

        if not plot_SLonly:
            # Yellow patch for regression range
            rect = patches.Rectangle((0.95, self.Results.proximalchannel),
                                     0.05,
                                     self.Results.distalchannel - self.Results.proximalchannel,
                                     linewidth=0, facecolor='yellow', alpha=0.5)
            ax.add_patch(rect)

        ax.legend()

    def _plot_freqran(self, ax):
        yl = ax.get_ylim()
        for f in [self.Results.startinglowfreq, self.Results.endinglowfreq]:
            ax.vlines(f, yl[0], yl[1], colors='b', linestyles='--')
        for f in [self.Results.startinghighfreq, self.Results.endinghighfreq]:
            ax.vlines(f, yl[0], yl[1], colors='r', linestyles='--')

    def _plot_physMarkers(self, ax, textpos, plot_SLonly):
        if not plot_SLonly:
            cross = self.Results.crossoverchannel
            low = self.Results.lowfreqmaxchannel
            high = self.Results.highfreqmaxchannel
        else:
            cross = self.Results.crossoverchannel - self.Results.proximalchannel
            low = self.Results.lowfreqmaxchannel - self.Results.proximalchannel
            high = self.Results.highfreqmaxchannel - self.Results.proximalchannel
            # Offset textpos for local plot if needed, but logic usually implies abs coords

        xlims = ax.get_xlim()

        ax.hlines(cross, xlims[0], xlims[1], colors='k', linestyles='-.', linewidth=0.5)
        ax.text(textpos, cross, 'Crossover', va='bottom', ha='right', fontsize=8)

        ax.hlines(low, xlims[0], xlims[1], colors='k', linestyles='-.', linewidth=0.5)
        ax.text(textpos, low, 'Alpha/Beta Peak', va='bottom', ha='right', fontsize=8)

        ax.hlines(high, xlims[0], xlims[1], colors='k', linestyles='-.', linewidth=0.5)
        ax.text(textpos, high, 'Gamma Peak', va='bottom', ha='right', fontsize=8)

    def plot_result(self, plot_SLonly=False):
        if self.Results is None: return

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 3)

        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_relpowMap(ax1, plot_SLonly)

        ax2 = fig.add_subplot(gs[0, 2], sharey=ax1)
        self.plot_bandedrelpow(ax2, plot_SLonly)

        t = f'G = {self.Results.goodnessvalue:.4f}, Omega = {self.Results.omega:.4f}'
        fig.suptitle(f'vFLIP2 Results: {t}')
        plt.tight_layout()
        plt.show()

