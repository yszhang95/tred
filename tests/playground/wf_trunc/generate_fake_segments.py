import numpy as np
import h5py

with h5py.File("trk2.hdf5", "w") as fh5:
    npdtype = {
        "fields"  : ["t0", "t0_start", "t0_end", "x_start", "y_start", "z_start", "x_end", "y_end", "z_end", "dE", "dEdx", "event_id", "pdg_id", "vertex_id"],
        "dtype"  : ["<f8", "<f8", "<f8", "<f4", "<f4", "<f4", "<f4", "<f4", "<f4", "<f4", "<f4", "<f4", "<u4", "<i4", "<u4"]
    }



    segments = np.array(
        [
            (0,0,0, 3.3,0.2,4.8, 3.3,0.2,5.0, 1.,2.1, 0, 211, 1000),
            (0,0,0, 4.0,0.2,4.8, 4.0,0.2,5.0, 1.,2.1, 0, 211, 1000),
        ],
        dtype = np.dtype([(n,d) for n,d in zip(npdtype['fields'], npdtype['dtype'])])
    )

    fh5.create_dataset('segments', data=segments)
