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
# License version 3 without disclosing the source code of your own applications.
#

import numpy as np
from collections import OrderedDict

import h5py
import mne
from mne.utils import logger
from mne.externals.h5io import write_hdf5

from ...markers.base import BaseMarkerSandbox, _read_container


class WeightedPhaseLagIndex(BaseMarkerSandbox):
    def __init__(self, tmin=None, tmax=None, fmin=None, fmax=None,
                 method_params=None, n_jobs='auto', comment='default'):
        BaseMarkerSandbox.__init__(
            self, tmin=None, tmax=None, comment=comment)
        if method_params is None:
            method_params = {}
        if fmax is None:
            fmax = np.inf
        self.fmin = fmin
        self.fmax = fmax
        self.method_params = method_params
        if n_jobs == 'auto':
            try:
                import multiprocessing as mp
                import mkl
                n_jobs = int(mp.cpu_count() / mkl.get_max_threads())
                logger.info(
                    'Autodetected number of jobs {}'.format(n_jobs))
            except:
                logger.info('Cannot autodetect number of jobs')
                n_jobs = 1
        self.n_jobs = n_jobs

    @property
    def _axis_map(self):
        return OrderedDict([
            ('channels', 0),
            ('channels_y', 1)
        ])

    def _fit(self, epochs):
        data, freqs, times, n_epochs, n_tappers = \
            mne.connectivity.spectral_connectivity(
                epochs, method='wpli', indices=None, 
                sfreq=epochs.info['sfreq'], mode='multitaper', fmin=self.fmin,
                fmax=self.fmax, tmin=self.tmin, tmax=self.tmax, faverage=True,
                n_jobs=self.n_jobs)
        self.data_ = np.squeeze(data)
        self.freqs_ = freqs[0]
        self.times_ = times
        self.n_epochs_ = n_epochs
        self.n_tappers_ = n_tappers

    @classmethod
    def _read(cls, fname, comment='default'):
        return _read_wpli(cls, fname=fname, comment=comment)


def _read_wpli(cls, fname, comment='default'):
    out = _read_container(cls, fname, comment=comment)
    return out


def read_wpli(fname, comment='default'):
    out = WeightedPhaseLagIndex._read(fname, comment=comment)
    return out
