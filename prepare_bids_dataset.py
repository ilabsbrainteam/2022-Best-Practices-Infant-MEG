from pathlib import Path
import shutil
import mne
from mne_bids import (BIDSPath, get_anat_landmarks, print_dir_tree, write_anat,
                      write_raw_bids, make_dataset_description)

this_path = Path(__file__).parent
task = "amnoise"
bids_root = this_path / f'{task}-bids'
bids_root.mkdir(exist_ok=True)
old_subjects_dir = this_path / 'subjects'
subjects = ['102']
event_id = {"auditory": 1}
print_dir_tree(bids_root, max_depth=3)
datatype = "meg"
bids_path = BIDSPath(root=bids_root, datatype=datatype)
subjects_dir = bids_path / 'derivatives' / 'freesurfer' / 'subjects'
subjects_dir.make_dirs()

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

{0}
""".format('\n'.join(refs))
with open(bids_path.root / 'README', 'w', encoding='utf-8-sig') as fid:
    fid.write(README)
raise RuntimeError

suffix = "meg"
for subject in subjects:
    bids_path = BIDSPath(
        subject=subject,
        task=task,
        suffix=suffix,
        datatype=datatype,
        root=bids_root,
    )
    old_subject = f'fc_12mo_{subject}'

    # Raw data
    raw = mne.io.read_raw_fif(
        this_path / f'{old_subject}_raw.fif', allow_maxshield='yes')
    events = mne.find_events(raw)
    events = events[events[:, 2] == 1]  # trim to useful events (all the same)
    assert len(events) == 110, len(events)
    assert raw.info["line_freq"] == 60
    write_raw_bids(
        raw, bids_path, events_data=events, event_id=event_id, overwrite=True,
        verbose=True)

    # Empty-room
    erm = mne.io.read_raw_fif(
        this_path / f'{old_subject}_erm_raw.fif', allow_maxshield='yes')
    assert erm.info["line_freq"] == 60
    er_date = erm.info["meas_date"].strftime("%Y%m%d")
    er_bids_path = BIDSPath(
        subject="emptyroom", session=er_date, task="noise", root=bids_root)
    write_raw_bids(erm, er_bids_path, overwrite=True, verbose=True)

    # MRI
    fs_subject = f'sub-{subject}'
    t1_fname = subjects_dir / fs_subject / 'mri' / 'T1.mgz'
    if not t1_fname.exists():
        # Need to copy over from old name
        shutil.copytree(old_subjects_dir / 'ANTS6-0Months3T',
                        subjects_dir / 'ANTS6-0Months3T')
        config = mne.coreg.read_mri_cfg(old_subject, old_subjects_dir)
        assert config.pop('n_params') == 3
        assert config['subject_from'] == 'ANTS6-0Months3T'
        print(f'+Copying MRI (rescaling {config["subject_from"]} to '
              f'{fs_subject})')
        mne.coreg.scale_mri(subject_to=fs_subject, subjects_dir=subjects_dir,
                            labels=False, annot=False, verbose=True,
                            **config)
        sol_file = (subjects_dir / fs_subject / 'bem' /
                    f'{fs_subject}-5120-5120-5120-bem-sol.fif')
        print('+Computing BEM solution')
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
        verbose=True)
