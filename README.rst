2022-Best-Practices-Infant-MEG
==============================

This repository contains a sample infant dataset analysis script. The
corresponding data files (to be downloaded to the same folder as
``pipeline.py``) can be obtained from OSF.io:

- `ct_sparse.fif <https://osf.io/h3xn7/download>`__
- `sss_cal.dat <https://osf.io/ryg5k/download>`__
- `fc_12mo_102_raw.fif <https://osf.io/x95yf/download>`__
- `fc_12mo_102_erm_raw.fif <https://osf.io/h5vsn/download>`__

This script requrires:

- `MNE-Python <https://mne.tools/dev>`__
- `h5io <https://github.com/h5io/h5io>`__
- `mnefun <https://github.com/LABSN/mnefun>`__

If you install `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`__,
you can set up an isolated environment with all dependencies with:

.. code-block:: console

    (base) $ conda create -n isolated -c conda-forge mne h5io
    (isolated) $ conda activate isolated
    (isolated) $ pip install https://github.com/LABSN/mnefun/zipball/dde46d7f58df8eccc55e3c56370c0bc27c7d1653

Then running ``python -i pipeline.py`` should work!

We plan to incorporate the mnefun functions and the workflow from
thes script into the MNE-Python and
`MNE-BIDS-Pipeline <https://mne.tools/mne-bids-pipeline/>`__ packages, see:

https://chanzuckerberg.com/eoss/proposals/building-pediatric-and-clinical-data-pipelines-for-mne-python/
