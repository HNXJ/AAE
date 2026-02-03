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

