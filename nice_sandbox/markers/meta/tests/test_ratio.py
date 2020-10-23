# NICE
# Copyright (C) 2017 - Authors of NICE-sandbox
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# You can be released from the requirements of the license by purchasing a
# commercial license. Buying such a license is mandatory as soon as you
# develop commercial activities as mentioned in the GNU Affero General Public
# License version 3 without disclosing the source code of your own
# applications.

import numpy as np

from numpy.testing import assert_array_equal

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
