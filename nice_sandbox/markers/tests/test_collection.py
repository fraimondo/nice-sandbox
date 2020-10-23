# NICE-Sandbox
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


from nose.tools import assert_equal, assert_true

from mne.utils import _TempDir

# our imports
from nice.markers import PowerSpectralDensity
from nice.markers import ContingentNegativeVariation
from nice.markers import PermutationEntropy
from nice.markers import PowerSpectralDensityEstimator

from nice_sandbox.markers.connectivity import WeightedPhaseLagIndex

from nice import Markers, read_markers

from nice.tests.test_collection import _get_data


def test_collecting_feature():
    """Test computation of spectral markers"""
    epochs = _get_data()[:2]
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto',
                       nperseg=128)
    estimator = PowerSpectralDensityEstimator(
        tmin=None, tmax=None, fmin=1., fmax=45., psd_method='welch',
        psd_params=psds_params, comment='default'
    )

    wpli = WeightedPhaseLagIndex()
    markers_list = [
        PowerSpectralDensity(estimator=estimator, fmin=1, fmax=4),
        ContingentNegativeVariation(),
        wpli
    ]

    markers = Markers(markers_list)
    # check states and names
    for name, marker in markers.items():
        assert_true(not any(k.endswith('_') for k in vars(marker)))
        assert_equal(name, marker._get_title())

    # check order
    assert_equal(list(markers.values()), markers_list)

    # check fit
    markers.fit(epochs)
    for t_marker in markers_list:
        assert_true(any(k.endswith('_') for k in vars(t_marker)))

    tmp = _TempDir()
    tmp_fname = tmp + '/test-smarkers.hdf5'
    markers.save(tmp_fname)
    markers2 = read_markers(tmp_fname)
    for ((k1, v1), (k2, v2)) in zip(markers.items(), markers2.items()):
        assert_equal(k1, k2)
        assert_equal(
            {k: v for k, v in vars(v1).items() if not k.endswith('_') and
             not k == 'estimator'},
            {k: v for k, v in vars(v2).items() if not k.endswith('_') and
             not k == 'estimator'})
    pe = PermutationEntropy().fit(epochs)
    markers._add_marker(pe)

    tmp = _TempDir()
    tmp_fname = tmp + '/test-markers.hdf5'
    markers.save(tmp_fname)
    markers3 = read_markers(tmp_fname)
    assert_true(pe._get_title() in markers3)

    assert_true(wpli._get_title() in markers3)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
