import uuid
import h5py
import numpy as np
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.ecephys import ElectricalSeries
from hdmf.data_utils import DataChunkIterator
from pynwb.file import Subject
from pynwb.misc import Units

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
        sst_dset = f.get('session_start_time')
        sst_str = _safe_decode(sst_dset[()]) if sst_dset is not None else None
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
                    desc_str = "N/A"
                    if 'description' in elec_dset[col].attrs:
                        desc_str = _safe_decode(elec_dset[col].attrs['description'])
                    nwbfile.add_electrode_column(name=col, description=desc_str)

            for i in range(len(ids)):
                eg_name = 'default'
                try: 
                    group_ref = elec_dset['group'][i]
                    if isinstance(group_ref, h5py.Reference):
                        eg_name = f[group_ref].name.split('/')[-1]
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
                        desc_str = "N/A"
                        if 'description' in grp[col].attrs:
                             desc_str = _safe_decode(grp[col].attrs['description'])
                        nwbfile.add_unit_column(name=col, description=desc_str)

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
                desc_str = f"Reconstructed {mod_key}"
                proc_mod = nwbfile.create_processing_module(name=mod_key, description=desc_str)
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
