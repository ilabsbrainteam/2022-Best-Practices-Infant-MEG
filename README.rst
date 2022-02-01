2022-Best-Practices-Infant-MEG
==============================

This repository contains a sample infant dataset analysis script. The
corresponding data files (to be downloaded to the same folder as
``pipeline.py``) can be obtained from `OSF.io <https://osf.io/2xf4y/files>`__.

This script requrires:

- `MNE-Python <https://mne.tools/dev>`__
- `h5io <https://github.com/h5io/h5io>`__
- `mnefun <https://github.com/LABSN/mnefun>`__

If you install Anaconda, you can get all dependencies with something like:

.. code-block:: console

    (base) $ conda create -n isolated -c conda-forge mne h5io
    (isolated) $ conda activate isolated
    (isolated) $ pip install https://github.com/LABSN/mnefun/zipball/dde46d7f58df8eccc55e3c56370c0bc27c7d1653

Then running ``python -i pipeline.py`` should work!

We plan to incorporate the mnefun functions and the workflow from
thes script into the MNE-Python and
`MNE-BIDS-Pipeline <https://mne.tools/mne-bids-pipeline/>`__ packages, see:

https://chanzuckerberg.com/eoss/proposals/building-pediatric-and-clinical-data-pipelines-for-mne-python/
