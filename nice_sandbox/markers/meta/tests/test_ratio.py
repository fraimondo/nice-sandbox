import numpy as np

from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal)

import functools

import mne

from nice.utils import create_mock_data_egi
from nice.markers.tests.test_markers import _base_io_test
from nice.markers.spectral import (PowerSpectralDensity,
                                    PowerSpectralDensityEstimator)

from nice_sandbox.markers.meta import Ratio, read_ratio

n_epochs = 30
raw = create_mock_data_egi(6, n_epochs * 386, stim=True)

triggers = np.arange(50, n_epochs * 386, 386)

raw._data[-1].fill(0.0)
raw._data[-1, triggers] = [10] * int(n_epochs / 2) + [20] * int(n_epochs / 2)

events = mne.find_events(raw)
event_id = {
    'foo': 10,
    'bar': 20,
}
epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=1.34,
                    preload=True, reject=None, picks=None,
                    baseline=(None, 0), verbose=False)
epochs.drop_channels(['STI 014'])
picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                       stim=False, exclude='bads')


def test_ratio():
    """Test computation of Ratio markers"""
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto',
                       nperseg=128)
    estimator = PowerSpectralDensityEstimator(
        tmin=None, tmax=None, fmin=1., fmax=45., psd_method='welch',
        psd_params=psds_params, comment='default'
    )
    psd1 = PowerSpectralDensity(estimator, fmin=1., fmax=4., comment='delta')
    psd2 = PowerSpectralDensity(estimator, fmin=4., fmax=8., comment='theta')
    markers = {
        psd1._get_title(): psd1,
        psd2._get_title(): psd2
    }
    ratio = Ratio(numerator=psd1, denominator=psd2,
                  comment='delta_theta')

    _base_io_test(ratio, epochs,
                  functools.partial(read_ratio, markers=markers,
                                    comment='delta_theta'))

    data1 = psd1.data_
    data2 = psd2.data_

    div_data = data1/data2

    assert_array_equal(div_data, ratio.data_)

    red_ratio = [{'axis': 'frequency', 'function': np.sum},
                 {'axis': 'epochs', 'function': np.mean},
                 {'axis': 'channels', 'function': np.mean}]

    topos_ratio = ratio.reduce_to_topo(reduction_func=red_ratio)

    freq_ax = psd1._axis_map['frequency']
    epochs_ax = psd1._axis_map['epochs']
    data1 = psd1._prepare_data(target='scalar', picks=None)
    data2 = psd2._prepare_data(target='scalar', picks=None)
    data1 = data1.sum(axis=freq_ax)
    data2 = data2.sum(axis=freq_ax)
    div_data = data1/data2
    topos_div = div_data.mean(axis=epochs_ax)

    assert_array_equal(topos_div, topos_ratio)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
