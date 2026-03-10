import numpy as np

def get_signal_array(nwb_file, event_timestamps, time_pre, time_post, signal_mode='lfp', probe_id=0, eye_dimension_index=0):
    """
    Extracts signal segments aligned to event timestamps.
    """
    signal_data_h5 = None
    signal_timestamps_h5 = None
    num_channels = 1
    signal_name = ""

    if signal_mode == 'lfp':
        signal_name = f'probe_{probe_id}_lfp'
        if signal_name in nwb_file.acquisition:
            electrical_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = electrical_series.data 
            signal_timestamps_h5 = electrical_series.timestamps 
            num_channels = signal_data_h5.shape[1] if signal_data_h5.ndim > 1 else 1
        else:
            print(f"Error: LFP data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'muae':
        signal_name = f'probe_{probe_id}_muae'
        if signal_name in nwb_file.acquisition:
            electrical_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = electrical_series.data 
            signal_timestamps_h5 = electrical_series.timestamps 
            num_channels = signal_data_h5.shape[1] if signal_data_h5.ndim > 1 else 1
        else:
            print(f"Error: MUAe data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'pupil':
        signal_name = 'pupil_1_tracking'
        if signal_name in nwb_file.acquisition:
            time_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = time_series.data 
            signal_timestamps_h5 = time_series.timestamps 
            num_channels = 1
        else:
            print(f"Error: Pupil tracking data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'eye':
        signal_name = 'eye_1_tracking'
        if signal_name in nwb_file.acquisition:
            time_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = time_series.data 
            signal_timestamps_h5 = time_series.timestamps 
            num_channels = 1
        else:
            print(f"Error: Eye tracking data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'photodiode':
        signal_name = 'photodiode_1_tracking'
        if signal_name in nwb_file.acquisition:
            time_series = nwb_file.acquisition[signal_name]
            signal_data_h5 = time_series.data 
            signal_timestamps_h5 = time_series.timestamps 
            num_channels = 1
        else:
            print(f"Error: Photodiode tracking data for {signal_name} not found.")
            return np.array([])
    elif signal_mode == 'convolved_spike_train':
        signal_name = 'convolved_spike_train_data'
        if 'convolved_spike_train' in nwb_file.processing and signal_name in nwb_file.processing['convolved_spike_train'].data_interfaces:
            time_series = nwb_file.processing['convolved_spike_train'].data_interfaces[signal_name]
            signal_data_h5 = time_series.data 
            signal_timestamps_h5 = time_series.timestamps 
            num_channels = signal_data_h5.shape[1] if signal_data_h5.ndim > 1 else 1
        else:
            print(f"Error: Convolved spike train data for {signal_name} not found.")
            return np.array([])
    else:
        print(f"Error: Invalid signal_mode '{signal_mode}'.")
        return np.array([])

    if signal_data_h5 is None or signal_timestamps_h5 is None or len(signal_timestamps_h5) == 0:
        return np.array([])

    signal_timestamps = signal_timestamps_h5[:]

    if len(signal_timestamps) > 1:
        sampling_rate = 1 / np.mean(np.diff(signal_timestamps))
    else:
        sampling_rate = 1000.0

    num_time_points_in_window = int(np.round((time_pre + time_post) * sampling_rate))

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
        data_segment_end_idx = min(len(signal_timestamps), end_idx)

        actual_samples_from_signal = data_segment_end_idx - data_segment_start_idx
        if actual_samples_from_signal <= 0: continue

        ideal_paste_start = int(np.round((signal_timestamps[data_segment_start_idx] - window_start_time) * sampling_rate))
        ideal_paste_end = ideal_paste_start + actual_samples_from_signal

        final_result_slice_start = max(0, ideal_paste_start)
        final_result_slice_end = min(num_time_points_in_window, ideal_paste_end)

        source_data_start_offset = final_result_slice_start - ideal_paste_start
        source_data_end_offset = source_data_start_offset + (final_result_slice_end - final_result_slice_start)

        if signal_mode == 'eye' and signal_data_h5.ndim > 1:
            current_data_segment = signal_data_h5[data_segment_start_idx:data_segment_end_idx, eye_dimension_index]
        else:
            current_data_segment = signal_data_h5[data_segment_start_idx:data_segment_end_idx]

        segment_to_copy = current_data_segment[source_data_start_offset : source_data_end_offset]

        if num_channels > 1 and signal_mode not in ['pupil', 'eye', 'photodiode']:
            result_array[i, final_result_slice_start:final_result_slice_end, :] = segment_to_copy
        else:
            result_array[i, final_result_slice_start:final_result_slice_end] = segment_to_copy

    return result_array
