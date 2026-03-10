def get_neuron_info(nwb, unit_id):
    """
    Retrieves info for a specific neuron by ID.
    """
    if nwb.units is None:
        return None, unit_id, None, None, None

    all_ids = nwb.units.id[:]

    try:
        index = list(all_ids).index(unit_id)
    except ValueError:
        return None, unit_id, None, None, None

    def get_col_val(col_name, idx):
        if col_name in nwb.units.colnames:
            return nwb.units[col_name][idx]
        return float('nan')

    peak_channel = get_col_val('peak_channel_id', index)
    snr = get_col_val('snr', index)
    presence_ratio = get_col_val('presence_ratio', index)

    area = "unknown"
    if nwb.electrodes is not None:
        try:
            elec_id = int(float(peak_channel))
            elec_ids = nwb.electrodes.id[:]
            if elec_id in elec_ids:
                elec_idx = list(elec_ids).index(elec_id)
                if 'location' in nwb.electrodes.colnames:
                    val = nwb.electrodes['location'][elec_idx]
                    area = val.decode('utf-8') if isinstance(val, bytes) else str(val)
                elif 'label' in nwb.electrodes.colnames:
                    val = nwb.electrodes['label'][elec_idx]
                    area = val.decode('utf-8') if isinstance(val, bytes) else str(val)
        except:
            pass

    return peak_channel, unit_id, snr, presence_ratio, area
