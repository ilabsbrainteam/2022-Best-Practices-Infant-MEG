"""Sample infant dataset analysis pipeline script."""

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import h5io
import mne
import mnefun

mne.set_log_level('warning')

# Input files
subject = 'fc_12mo_102'
manual_bads = [  # manually identified flux jumps via visual inspection
    'MEG0142', 'MEG0313', 'MEG0422', 'MEG1223', 'MEG1442', 'MEG1443',
    'MEG2623']

raw_fname = f'{subject}_raw.fif'
erm_fname = f'{subject}_erm_raw.fif'
cal_fname = 'sss_cal.dat'
ct_fname = 'ct_sparse.fif'

# Output files
bad_fname = f'{subject}-bads.txt'
raw_sss_fname = f'{subject}_raw_sss.fif'
raw_sss_ssp_fname = f'{subject}_proj_raw_sss.fif'
epochs_fname = f'{subject}-epo.fif'
evoked_fname = f'{subject}-ave.fif'
cov_fname = f'{subject}-cov.fif'
surrogate = 'ANTS6-0Months3T'
subjects_dir = 'subjects'
os.makedirs('subjects', exist_ok=True)
trans_fname = f'{subject}-trans.fif'
bem_fname = op.join(
    subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')
fwd_fname = f'{subject}-fwd.fif'
inv_fname = f'{subject}-inv.fif'

int_order = 6
st_correlation = 0.98
st_duration = 10
regularize = 'in'
dist_limit = 0.01
gof_limit = 0.95
reject = dict(grad=1500e-13, mag=6000e-15)

###############################################################################
# Load raw data

raw = mne.io.read_raw_fif(raw_fname, allow_maxshield='yes')
raw.fix_mag_coil_types()
erm = mne.io.read_raw_fif(erm_fname, allow_maxshield='yes')
erm.fix_mag_coil_types()
R, head_origin = mne.bem.fit_sphere_to_headshape(
    raw.info, units='m', verbose=False)[:2]

###############################################################################
# Automatically determine bad channels, see:
# https://mne.tools/stable/auto_tutorials/preprocessing/60_maxwell_filtering_sss.html

if not op.isfile(bad_fname):
    print('Computing bad channels ...', end='')
    all_bads = set(manual_bads)
    for r in (raw, erm):
        r = raw.copy().load_data()
        if r.info['dev_head_t'] is None:
            coord_frame, origin = 'meg', (0., 0., 0.)
        else:
            coord_frame, origin = 'head', head_origin
        mne.chpi.filter_chpi(r, allow_line_only=True, t_window='auto')
        r.info['bads'] = manual_bads
        bads, flats = mne.preprocessing.find_bad_channels_maxwell(
            r, 7, origin=origin, coord_frame=coord_frame,
            bad_condition='warning', calibration=cal_fname,
            cross_talk=ct_fname, h_freq=None)
        all_bads = all_bads.union(set(bads + flats))
    all_bads = sorted(all_bads)
    with open(bad_fname, 'w') as fid:
        for ch in all_bads:
            fid.write(f'{ch}\n')
    print(f' bad channels: {all_bads}')
with open(bad_fname, 'r') as fid:
    bads = fid.read().strip().split('\n')
raw.info['bads'] = bads
erm.info['bads'] = bads


###############################################################################
# Compute head position as a function of time, see
# https://mne.tools/stable/auto_tutorials/preprocessing/59_head_positions.html

locs_fname = raw.filenames[0][:-4] + '-chpi_locs.h5'
count_fname = raw.filenames[0][:-4] + '-chpi_counts.h5'
pos_fname = raw.filenames[0][:-4] + '.pos'
if not op.isfile(locs_fname):
    print('Computing cHPI amplitudes and locations ...')
    # Under the hood, this function calls:
    # chpi_amps = mne.chpi.compute_chpi_amplitudes(raw, t_window='auto')
    # chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amps)
    fit_t, counts, n_coils, chpi_locs = mnefun.compute_good_coils(
        raw, t_window='auto', dist_limit=dist_limit, gof_limit=gof_limit,
        verbose=True)
    h5io.write_hdf5(locs_fname, chpi_locs, title='mnepython')
    h5io.write_hdf5(
        count_fname, dict(fit_t=fit_t, counts=counts, n_coils=n_coils),
        title='mnepython')
    del fit_t, counts, n_coils, chpi_locs

if not op.isfile(pos_fname):
    print('Computing head position ...')
    chpi_locs = h5io.read_hdf5(locs_fname, title='mnepython')
    head_pos = mne.chpi.compute_head_pos(
        raw.info, chpi_locs, dist_limit=dist_limit, gof_limit=gof_limit)
    mne.chpi.write_head_pos(pos_fname, head_pos)
    del chpi_locs, head_pos

###############################################################################
# Apply tSSS with movement compensation to time-weighted average head pos, see
# https://mne.tools/stable/auto_examples/preprocessing/movement_compensation.html
# https://mne.tools/stable/auto_examples/preprocessing/movement_detection.html

if not op.isfile(raw_sss_fname):
    print('Applying tSSS with movement compensation to '
          'time-weighted average head position ...')
    r = raw.copy().load_data()
    mne.chpi.filter_chpi(r, t_window='auto', verbose=True)
    # Lowest HPI frequency is 83 Hz, so make sure it is well suppressed
    # (there is leakage slightly lower than this because the signals are
    # modulated by movement, so we filter slightly lower)
    r.filter(None, 75, h_trans_bandwidth=5)
    e = erm.copy().load_data()
    e.filter(None, 40)
    e.del_proj()
    proj = mne.compute_proj_raw(
        e, n_mag=3, n_grad=3, meg='combined', verbose=True)
    head_pos = mne.chpi.read_head_pos(pos_fname)
    destination = mne.preprocessing.compute_average_dev_head_t(raw, head_pos)
    raw_sss = mne.preprocessing.maxwell_filter(
        r, head_pos=head_pos,
        origin=head_origin, int_order=int_order,
        st_correlation=st_correlation, st_duration=st_duration,
        calibration=cal_fname, cross_talk=ct_fname, coord_frame='head',
        bad_condition='warning', regularize=regularize,
        destination=destination, extended_proj=proj, verbose=True)
    raw_sss.save(raw_sss_fname)
    del raw_sss, head_pos

###############################################################################
# Compute and apply SSP and filtering, see
# https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html

if not op.isfile(raw_sss_ssp_fname):
    print('Computing ECG SSP ...', end='')
    raw_sss = mne.io.read_raw_fif(raw_sss_fname).del_proj()
    ecg_events, _, _ = mne.preprocessing.find_ecg_events(
        raw_sss, qrs_threshold=0.1)
    bpm = len(ecg_events) / raw.times[-1] * 60
    print(f' heart rate: {bpm:0.1f} bpm ...', end='')
    raw_ecg = raw_sss.copy().load_data().filter(1, 35)
    ecg_epochs = mne.Epochs(
        raw_ecg, ecg_events, tmin=-0.5, tmax=0.5, reject=reject,
        baseline=(None, None))
    ecg_evoked = ecg_epochs.average()
    assert ecg_evoked.nave > 0.8 * len(ecg_events)  # not too many removed
    proj = mne.compute_proj_evoked(
        ecg_evoked, n_mag=3, n_grad=3, meg='combined')
    print(' variance explained: '
          f'{100 * sum(p["explained_var"] for p in proj):0.1f}%')
    raw_sss.add_proj(proj, remove_existing=True)
    raw_sss.save(raw_sss_ssp_fname)
    del raw_sss

###############################################################################
# Epoch and downsample, see
# https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html

AM_onset, AM_duration = 0.3, 6.
tmax = AM_onset + AM_duration + 0.5
if not op.isfile(epochs_fname):
    print('Computing epochs ...')
    r = mne.io.read_raw_fif(raw_sss_ssp_fname).load_data()
    events = mne.find_events(r)
    events = events[events[:, 2] == 1]  # trim to useful events (all the same)
    assert len(events) == 110, len(events)
    decim = int(round(r.info['sfreq'])) // 240  # destination sample rate
    epochs = mne.Epochs(r, events, event_id=dict(Auditory=1), tmax=tmax,
                        reject=reject, decim=decim, preload=True, proj=True)
    assert 60 <= len(epochs) < 110, len(epochs)  # not too many rejected
    epochs.save(epochs_fname)
    del events, epochs

###############################################################################
# Compute average, see
# https://mne.tools/stable/auto_tutorials/evoked/10_evoked_overview.html

if not op.isfile(evoked_fname):
    print('Computing evoked ...')
    mne.read_epochs(epochs_fname).average().save(evoked_fname)

###############################################################################
# Compute covariance (regularized), see
# https://mne.tools/stable/auto_tutorials/forward/90_compute_covariance.html

if not op.isfile(cov_fname):
    print('Computing covariance ...', end='')
    epochs = mne.read_epochs(epochs_fname, preload=True)
    rank = mne.compute_rank(epochs, tol=1e-5, tol_kind='relative')
    cov = mne.compute_covariance(
        epochs, tmax=0, method='shrunk', rank=rank, verbose='error')
    print(f' rank: {rank["meg"]} ...')
    mne.write_cov(cov_fname, cov)
    del epochs, rank, cov
cov = mne.read_cov(cov_fname)

###############################################################################
# Create surrogate MRI, see
# https://mne.tools/stable/auto_tutorials/forward/25_automated_coreg.html
# This MRI already includes a 3-layer BEM and surface source space suitable
# for inverse imaging, so we need to rescale it to our subject's digitization.

info = mne.io.read_info(raw_sss_fname)
if not op.isdir(op.join(subjects_dir, subject)):
    print('Creating surrogate MRI subject ...')
    mne.datasets.fetch_infant_template('6mo', subjects_dir=subjects_dir)
    # Modify fiducial point to more accurately reflect our digitization
    fid_fname = op.join(
        subjects_dir, surrogate, 'bem', f'{surrogate}-fiducials.fif')
    dig = mne.channels.read_dig_fif(fid_fname)
    dig.dig[1]['r'] = np.array([0, 0.071, 0])
    coreg = mne.coreg.Coregistration(info, surrogate, subjects_dir, dig.dig)
    coreg.set_scale_mode('uniform').set_fid_match('matched')
    coreg.fit_fiducials()
    coreg.set_scale_mode('3-axis')
    for _ in range(4):  # in lieu of setting a tolerance
        coreg.fit_icp(nasion_weight=1)
    mne.write_trans(trans_fname, coreg.trans, overwrite=True)
    mne.coreg.scale_mri(
        surrogate, subject, coreg.scale, subjects_dir=subjects_dir,
        annot=True)
    bem = mne.read_bem_surfaces(f'{bem_fname[:-12]}-bem.fif')
    bem = mne.make_bem_solution(bem)
    mne.write_bem_solution(bem_fname, bem)
    del bem

###############################################################################
# Compute forward and inverse
# https://mne.tools/stable/auto_tutorials/forward/30_forward.html
# https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html

if not op.isfile(fwd_fname):
    src_fname = op.join(subjects_dir, subject, 'bem',
                        f'{subject}-oct-6-src.fif')
    fwd = mne.make_forward_solution(info, trans_fname, src_fname, bem_fname)
    mne.write_forward_solution(fwd_fname, fwd)
    del fwd, src_fname

if not op.isfile(inv_fname):
    fwd = mne.read_forward_solution(fwd_fname)
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=1.)
    mne.minimum_norm.write_inverse_operator(inv_fname, inv)

###############################################################################
# Apply inverse
# https://mne.tools/stable/auto_examples/inverse/vector_mne_solution.html

lambda2, method = 1. / 9., 'dSPM'
evoked = mne.read_evokeds(evoked_fname)[0]
inv = mne.minimum_norm.read_inverse_operator(inv_fname)
stc = mne.minimum_norm.apply_inverse(
    evoked, inv, lambda2, method, pick_ori='vector')
labels = mne.read_labels_from_annot(  # Heschl's gyrus
    subject, 'aparc.a2009s', hemi='lh', regexp='G_temp_sup-G_T_transv',
    subjects_dir=subjects_dir)
assert len(labels) == 1  # the auditory label
label = mne.extract_label_time_course(stc, labels, src=inv['src'])

###############################################################################
# Compute TFR of average and peak auditory response vertex
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html

sfreq = evoked.info['sfreq']
idx_zero = np.where(evoked.times >= AM_onset)[0][0]
stim_t = np.arange(0, 78 / 12.9630, 1 / sfreq)
fmin, fmax, fstep = 2, 80, 0.5
freqs = np.arange(fmin, fmax + 1e-5, fstep)
xm = (fmin + (fmax - fmin) * stim_t / stim_t[-1])
am = np.sin(
    2 * np.pi * (np.cumsum(xm) / sfreq) - np.pi / 2 - 2 / sfreq * 2 * np.pi)
am = np.concatenate((np.zeros(idx_zero), am))
am = np.concatenate((am, np.zeros(len(evoked.times) - am.size)))
n_cycles = freqs  # fix time-freq tiling
freqs = np.arange(fmin, fmax + 1e-5, fstep)
t_mask = (evoked.times >= AM_onset) & (evoked.times <= AM_onset + AM_duration)
del xm, stim_t, idx_zero, AM_duration

tfr_kwargs = dict(n_cycles=n_cycles, use_fft=True, freqs=freqs, n_jobs=4,
                  output='power', sfreq=sfreq)
stim_power = mne.time_frequency.tfr_array_morlet(
    am[np.newaxis, np.newaxis, :], **tfr_kwargs)[0, 0]
sensor_power = mne.time_frequency.tfr_array_morlet(
    evoked.data[np.newaxis], **tfr_kwargs)[0]
assert sensor_power.shape == (
    len(evoked.ch_names), len(freqs), len(evoked.times))
label_power = mne.time_frequency.tfr_array_morlet(
    label, **tfr_kwargs)[0]
shift = int(round(0.04 * evoked.info['sfreq']))
print(f'Compensating for minimal neural delay: '
      f'{1000 * shift / evoked.info["sfreq"]:0.3f} ms')


def baseline_correct_shift(power, shift):
    """Z-score the data using non-extreme, non-diag values."""
    baseline_mask = (power < np.mean(power, axis=-1, keepdims=True) +
                     3 * np.std(power, axis=-1, keepdims=True))
    # baseline_mask &= lag_mask
    z = power.copy()
    z[~baseline_mask] = np.nan
    power -= np.nanmean(z, axis=-1, keepdims=True)
    power /= np.nanstd(z, axis=-1, keepdims=True)
    power[..., :-shift] = power[..., shift:]
    power[..., -shift:] = power[..., [shift]]
    return power


n_f_t = t_mask.sum() * len(freqs)
sensor_power = baseline_correct_shift(sensor_power, shift=shift)
sensor_power_crop = sensor_power[..., t_mask]
sensor_power_crop = sensor_power_crop.reshape(len(evoked.ch_names), n_f_t)
label_power = baseline_correct_shift(label_power, shift=shift)
label_power_crop = label_power[..., t_mask]
label_power_crop = label_power_crop.reshape(3, n_f_t)
stim_power_crop = stim_power[:, t_mask].ravel()
sensor_corrs = np.corrcoef(stim_power_crop, sensor_power_crop)[0, 1:]
# to get the coefficient of multiple correlation for the label (which has
# three orientations), we orthogonalize then sum the variance squared
_, _, label_power_crop_orth = np.linalg.svd(
    label_power_crop, full_matrices=False)
label_corr = np.corrcoef(stim_power_crop, label_power_crop_orth)[0, 1:]
# sqrt of sum of squares (variances explained) gives us the effective R value
label_corr = np.linalg.norm(label_corr)

###############################################################################
# Plot TFRs

t = evoked.times
mag_idx = np.argmax(sensor_corrs[2::3]) * 3 + 2
fig, axes = plt.subplots(1, 3, figsize=(7, 3), constrained_layout=True)
pc_kwargs = dict(zorder=4, cmap='viridis')
ti_kwargs = dict(fontsize=10)
text_kwargs = dict(
    x=t[t_mask][0], y=freqs[-2], ha='left', va='top',
    color='w', fontweight='bold', size=8, zorder=6)
delta = 1. / sfreq / 2.
x = np.concatenate([[t[0] - delta], t + delta])
delta = fstep / 2.
y = np.concatenate([[freqs[0] - delta], freqs + delta])
axes[0].pcolor(x, y, stim_power, **pc_kwargs)
axes[0].set_title('Stimulus AM', **ti_kwargs)
axes[0].set_ylabel('Frequency (Hz)')
axes[1].pcolor(x, y, sensor_power[mag_idx], **pc_kwargs)
axes[1].set_title(f'Sensor TFR\n{evoked.ch_names[mag_idx]}', **ti_kwargs)
axes[1].text(s=f'R={sensor_corrs[mag_idx]:0.3f}', **text_kwargs)
axes[2].pcolor(x, y, np.linalg.norm(label_power, axis=0), **pc_kwargs)
axes[2].set_title(f'Source label TFR\n{labels[0].name}', **ti_kwargs)
axes[2].text(s=f'R={label_corr:0.3f}', **text_kwargs)
for ax in axes:
    ax.contour(t, freqs, stim_power, zorder=5, colors='pink', levels=[1.0],
               linewidths=[0.5], linestyles=['-'], alpha=1.0)
    ax.set(xlabel='Time (sec)',
           xlim=(t[0], t[-1]),
           ylim=(freqs[0], freqs[-1]))

###############################################################################
# Make a quality control report
# (https://mne.tools/stable/auto_tutorials/intro/70_report.html)

report = mne.Report(info_fname=evoked.info, title=f'{subject} QC Report',
                    raw_psd=True)
# cHPI coil SNR
# (time-varying estimate of cHPI signal quality)
fig_snr = mnefun.plot_chpi_snr_raw(raw, 0.2, verbose=True)
report.add_figure(fig_snr, title='cHPI SNR', tags=('head-movement',))
# Good coil count
# (time-varying quantification of problematic coil displacement)
data = h5io.read_hdf5(count_fname, title='mnepython')
fig_coils = mnefun.plot_good_coils(data)
report.add_figure(fig_coils, title='Good cHPI coils', tags=('head-movement',))
# Head positions
# (time-varying estimated head positions)
head_pos = mne.chpi.read_head_pos(pos_fname)
fig_head_pos = mne.viz.plot_head_positions(info=evoked.info, pos=head_pos)
report.add_figure(fig_head_pos, title='Head position', tags=('head-movement',))
# Raw data
r = raw.copy().load_data().filter(None, 40)
report.add_raw(r, title='Raw (filtered)')
del r
raw_sss = mne.io.read_raw_fif(
    raw_sss_fname).copy().load_data().filter(None, 40)
report.add_raw(raw_sss, title='Raw (SSS, filtered)')
del raw_sss
raw_ssp = mne.io.read_raw_fif(raw_sss_ssp_fname).load_data().apply_proj()
report.add_raw(raw_ssp, title='Raw (SSS, SSP, filtered)')
del raw_ssp
# Coreg
report.add_trans(subject=subject, info=evoked.info, title='Coregistration',
                 trans=trans_fname, subjects_dir=subjects_dir)
# Evoked
report.add_evokeds(evoked)
# STC
report.add_stc(stc.magnitude(), title='Source space',
               subjects_dir=subjects_dir)
# TFRs
report.add_figure(fig, title='TFRs', tags=('TFR', 'evoked', 'source-estimate'))
# Generate HTML (and view it)
report.save(f'{subject}-report.html', overwrite=True)
