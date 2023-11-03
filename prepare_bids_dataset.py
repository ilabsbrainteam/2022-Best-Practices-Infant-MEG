import os
from pathlib import Path
import shutil  # noqa

# TODO: Resolve https://github.com/mne-tools/mne-bids-pipeline/issues/581
# to allow for source-space analyses.

import mne
from mne_bids import (
    BIDSPath,
    get_anat_landmarks,
    print_dir_tree,
    write_anat,
    write_raw_bids,
    make_dataset_description,
    write_meg_calibration,
    write_meg_crosstalk,
)  # noqa

task = "amnoise"
datatype = suffix = "meg"

this_path = Path(__file__).parent
bids_root = this_path / f'{task}-bids'
bids_root.mkdir(exist_ok=True)
old_subjects_dir = this_path / 'subjects'
subjects = ['102']
event_id = {"auditory": 1}
subjects_dir = bids_root / 'derivatives' / 'freesurfer' / 'subjects'
os.makedirs(subjects_dir, exist_ok=True)

# Description of the dataset
refs = [
    'Mittag, M., Larson, E., Clarke, M., Taulu, S., & Kuhl, P. K. (2021). Auditory deficits in infants at risk for dyslexia during a linguistic sensitive period predict future language. NeuroImage: Clinical, 30, 102578. https://doi.org/10.1016/j.nicl.2021.102578',  # noqa: E501
    'Mittag, M., Larson, E., Taulu, S., Clarke, M., & Kuhl, P. K. (2022). Reduced Theta Sampling in Infants at Risk for Dyslexia across the Sensitive Period of Native Phoneme Learning. International Journal of Environmental Research and Public Health, 19(3), 1180. https://doi.org/10.3390/ijerph19031180',  # noqa: E501
    ]
make_dataset_description(
    path=bids_root, name=task,
    authors=['Maria Mittag', 'Eric Larson', 'Maggie Clarke', 'Samu Taulu', 'Patricia K. Kuhl'],  # noqa: E501
    how_to_acknowledge='If you use this data, please cite the references provided in this dataset description.',  # noqa: E501
    data_license='CC-BY-SA',
    ethics_approvals=['Human Subjects Division at the University of Washington'],  # noqa: E501
    references_and_links=refs,
    overwrite=True)
README = """\
ILABS amnoise MEG BIDS dataset
==============================

This dataset contains MEG data from a single infant subject. For more
information, see the following publications, which should be cited if you use
this data:

- {0}

The data were converted with MNE-BIDS:

- Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896). https://doi.org/10.21105/joss.01896
- Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A., Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J., Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data structure extended to magnetoencephalography. Scientific Data, 5, 180110. https://doi.org/10.1038/sdata.2018.110
""".format('\n- '.join(refs))
README_FS_DERIVATIVES = """\
FreeSurfer derivatives for source imaging
=========================================

These were produced using ``mne coreg`` and
``mne.datasets.fetch_infant_dataset`` on the ``ANTS6-0Months3T`` template.
"""

for subject in subjects:
    bids_path = BIDSPath(
        subject=subject,
        task=task,
        suffix=suffix,
        datatype=datatype,
        root=bids_root,
    )
    with open(bids_root / 'README', 'w') as fid:
        fid.write(README)
    with open(bids_root / 'derivatives' / 'freesurfer' / 'README', 'w') as fid:
        fid.write(README_FS_DERIVATIVES)
    old_subject = f'fc_12mo_{subject}'

    # Raw data
    raw = mne.io.read_raw_fif(
        this_path / f'{old_subject}_raw.fif', allow_maxshield='yes')
    events = mne.find_events(raw)
    events = events[events[:, 2] == 1]  # trim to useful events (all the same)
    assert len(events) == 110, len(events)
    assert raw.info["line_freq"] == 60
    write_raw_bids(
        raw, bids_path, events=events, event_id=event_id, overwrite=True,
        verbose=False)

    # Empty-room
    erm = mne.io.read_raw_fif(
        this_path / f'{old_subject}_erm_raw.fif', allow_maxshield='yes')
    assert erm.info["line_freq"] == 60
    er_date = erm.info["meas_date"].strftime("%Y%m%d")
    er_bids_path = BIDSPath(
        subject="emptyroom", session=er_date, task="noise", root=bids_root)
    write_raw_bids(erm, er_bids_path, overwrite=True, verbose=False)

    # Maxwell filter files
    write_meg_calibration(this_path / "sss_cal.dat", bids_path=bids_path)
    write_meg_crosstalk(this_path / "ct_sparse.fif", bids_path=bids_path)
    """
    # MRI
    fs_subject = f'sub-{subject}'
    t1_fname = subjects_dir / fs_subject / 'mri' / 'T1.mgz'
    if not t1_fname.exists():
        # Need to copy over from old name
        config = mne.coreg.read_mri_cfg(old_subject, old_subjects_dir)
        subject_from = config['subject_from']
        assert subject_from == 'ANTS6-0Months3T'
        subject_from_new_path = subjects_dir / subject_from
        shutil.copytree(old_subjects_dir / subject_from, subject_from_new_path)
        assert config.pop('n_params') == 3
        print(f'Copying MRI (rescaling {subject_from} to {fs_subject})')
        mne.coreg.scale_mri(subject_to=fs_subject, subjects_dir=subjects_dir,
                            labels=False, annot=False, verbose=True,
                            **config)
        shutil.rmtree(subject_from_new_path)
        sol_file = (subjects_dir / fs_subject / 'bem' /
                    f'{fs_subject}-5120-5120-5120-bem-sol.fif')
        print('Computing BEM solution')
        sol = mne.make_bem_solution(str(sol_file)[:-8] + '.fif')
        mne.write_bem_solution(sol_file, sol)
    # transformation matrix
    trans = mne.read_trans(this_path / f'{old_subject}-trans.fif')
    t1w_bids_path = BIDSPath(subject=subject, root=bids_root, suffix="T1w")
    landmarks = get_anat_landmarks(
        t1_fname, info=raw.info, trans=trans, fs_subject=fs_subject,
        fs_subjects_dir=subjects_dir)
    write_anat(
        image=t1_fname, bids_path=t1w_bids_path, landmarks=landmarks,
        overwrite=True, verbose=True)
    """
print_dir_tree(bids_root)
