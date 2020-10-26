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
from nice import Markers
from nice.markers.tests.test_markers import _base_io_test
from nice.markers.spectral import (PowerSpectralDensity,
                                   PowerSpectralDensityEstimator)

from nice_sandbox.markers.meta import Passthrough, read_passthrough

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


def test_passthrough():
    """Test computation of Passthrough markers"""
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto',
                       nperseg=128)
    estimator = PowerSpectralDensityEstimator(
        tmin=None, tmax=None, fmin=1., fmax=45., psd_method='welch',
        psd_params=psds_params, comment='default'
    )
    psd_delta = PowerSpectralDensity(
        estimator, fmin=1., fmax=4., comment='delta')

    markers = {
        psd_delta._get_title(): psd_delta,
    }

    ant_delta = Passthrough(parent=psd_delta, comment='anterior_delta')
    pos_delta = Passthrough(parent=psd_delta, comment='posterior_delta')

    _base_io_test(ant_delta, epochs,
                  functools.partial(read_passthrough, markers=markers,
                                    comment='anterior_delta'))

    data1 = ant_delta.data_
    data2 = pos_delta.data_
    # Data should be the same
    assert_array_equal(data1, data2)

    red_delta = [
        {'axis': 'frequency', 'function': np.sum},
        {'axis': 'epochs', 'function': np.mean},
        {'axis': 'channels', 'function': np.mean}]

    # If we apply the same reduction, we should get the same data
    ant_topos = ant_delta.reduce_to_topo(reduction_func=red_delta)
    pos_topos = pos_delta.reduce_to_topo(reduction_func=red_delta)

    assert_array_equal(ant_topos, pos_topos)

    # Now apply different picks
    ant_scalar = ant_delta.reduce_to_scalar(
        reduction_func=red_delta,
        picks={'channels': [0]}
    )

    pos_scalar = pos_delta.reduce_to_scalar(
        reduction_func=red_delta,
        picks={'channels': [1]}
    )

    m_list = [
        ant_delta,
        pos_delta
    ]
    mc = Markers(m_list)
    mc.fit(epochs)

    reduction_params = {}
    reduction_params['Passthrough/anterior_delta'] = {
        'reduction_func': red_delta,
        'picks': {'channels': [0]}
    }
    reduction_params['Passthrough/posterior_delta'] = {
        'reduction_func': red_delta,
        'picks': {'channels': [1]}
    }

    scalars = mc.reduce_to_scalar(reduction_params)

    assert ant_scalar == scalars[0]
    assert pos_scalar == scalars[1]


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
