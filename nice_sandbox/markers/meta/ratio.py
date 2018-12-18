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

import h5py
from mne.utils import logger
from mne.externals.h5io import write_hdf5

from ...markers.base import BaseMarkerSandbox, _read_container


class Ratio(BaseMarkerSandbox):
    def __init__(self, numerator=None, denominator=None, comment='defalt'):
        BaseMarkerSandbox.__init__(
            self, tmin=None, tmax=None, comment=comment)
        self.numerator = numerator
        self.denominator = denominator
        self._check()

    @property
    def _axis_map(self):
        return self.numerator._axis_map

    def _check(self):
        if self.numerator is not None and self.denominator is not None:
            # Check: axis_map should be the equal
            for (k1, v1), (k2, v2) in zip(self.numerator._axis_map.items(),
                                          self.denominator._axis_map.items()):
                if k1 != k2 or v1 != v2:
                    raise ValueError('Numerator and Denominator markers must '
                                     'have the same shape.')

    def _fit(self, epochs):
        if self.numerator is None:
            raise ValueError('Need a numerator to be able to fit')
        if self.denominator is None:
            raise ValueError('Need a denominator to be able to fit')

        self._check()

        if not hasattr(self.numerator, 'data_'):
            logger.warning(
                'Numerator not fit. If this is part of a feature collection, '
                'it should be placed after the corresponding estimator.')
            self.numerator.fit(epochs)
        if not hasattr(self.denominator, 'data_'):
            logger.warning(
                'Denominator not fit. If this is part of a feature collection, '
                'it should be placed after the corresponding estimator.')
            self.denominator.fit(epochs)

    @property
    def data_(self):
        logger.warning('This attribute should not be accessed directly '
                       'as it depends on the numerator/denominator data_ '
                       'attribute')
        return self.numerator.data_ / self.denominator.data_

    def _prepare_reduction(self, reduction_func, target, picks):
        data_numerator = self.numerator._prepare_data(picks, target)
        data_denominator = self.denominator._prepare_data(picks, target)
        _axis_map = self._axis_map
        funcs = list()
        axis = []
        axis_to_preserve = self._get_preserve_axis(target)
        if len(axis_to_preserve) > 0:
            removed_axis = []
            for this_axis_to_preserve in axis_to_preserve:
                removed_axis.append(_axis_map.pop(this_axis_to_preserve))
            if reduction_func is not None:
                reduction_func = [i for i in reduction_func
                                  if i['axis'] not in axis_to_preserve]
        permutation_list = list()
        if reduction_func is None:
            for remaining_axis in _axis_map.values():
                permutation_list.append(remaining_axis)
                funcs.append(np.mean)
                axis.append(remaining_axis)
        elif len(reduction_func) == len(_axis_map):
            for rec in reduction_func:
                this_axis = _axis_map.pop(rec['axis'])
                permutation_list.append(this_axis)
                funcs.append(rec['function'])
                axis.append(rec['axis'])
        else:
            raise ValueError('Run `python -c "import this"` to see '
                             'why we will not tolerate these things')
        if len(axis_to_preserve) > 0:
            permutation_list += removed_axis

        found = False
        for t_ax in axis:
            if t_ax in ['epochs', 'channels'] and found is False:
                found = True
            if found is True and t_ax not in ['epochs', 'channels']:
                raise ValueError('Cannot reduce Ratio marker with this axis '
                                 'order. Channels and Epochs should be the '
                                 'two last axis to reduce (in any order).')
        data_numerator = np.transpose(data_numerator, permutation_list)
        data_denominator = np.transpose(data_denominator, permutation_list)
        return (data_numerator, data_denominator), funcs, axis

    def _reduce_to(self, reduction_func, target, picks):
        """ Reduce ratio

        This reduction will work in a particular way. It will first reduce
        numerator and denominator separately until only epochs and channels
        dimensions are left.

        """
        if not hasattr(self, 'data_'):
            raise ValueError('You did not fit me. Do it again after fitting '
                             'some data!')
        (data_num, data_den), funcs, axis = self._prepare_reduction(
            reduction_func, target, picks)

        out = None
        for func, ax in zip(funcs, axis):
            if ax not in ['channels', 'epochs']:
                data_num = func(data_num, axis=0)
                data_den = func(data_den, axis=0)
            else:
                if out is None:
                    out = data_num / data_den
                out = func(out, axis=0)
        return out

    def save(self, fname, overwrite=False):
        self._save_info(fname, overwrite=overwrite)
        save_vars = self._get_save_vars(
            exclude=['data_', 'numerator', 'denominator', 'tmin', 'tmax'])

        numerator_name = self.numerator._get_title()
        denominator_name = self.denominator._get_title()

        save_vars['numerator_name_'] = numerator_name
        save_vars['denominator_name_'] = denominator_name

        has_numerator = False
        has_denominator = False

        with h5py.File(fname) as h5fid:
            if numerator_name in h5fid:
                has_numerator = True
                logger.info('Numerator already present in HDF5 file, '
                            'will not be overwritten')
            if denominator_name in h5fid:
                has_denominator = True
                logger.info('Denominator already present in HDF5 file, '
                            'will not be overwritten')

        if not has_numerator:
            logger.info('Writing numerator to HDF5 file')
            self.numerator.save(fname, overwrite=overwrite)
        if not has_denominator:
            logger.info('Writing denominator to HDF5 file')
            self.denominator.save(fname, overwrite=overwrite)

        write_hdf5(
            fname, save_vars, overwrite=overwrite,
            title=self._get_title(), slash='replace')

    @classmethod
    def _read(cls, fname, markers=None, comment='default'):
        return _read_ratio(cls, fname=fname, markers=markers, comment=comment)


def _read_ratio(cls, fname, markers=None, comment='default'):
    out = _read_container(cls, fname, comment=comment)
    if markers is None:
        # Get class of each estimator and read
        raise NotImplementedError(
            'This feature is not yet implemented. Please read all the '
            'markers first')

    out.numerator = markers[out.numerator_name_]
    out.denominator = markers[out.denominator_name_]
    del out.numerator_name_
    del out.denominator_name_
    return out


def read_ratio(fname, markers=None, comment='default'):
    out = Ratio._read(fname, markers=markers, comment=comment)
    return out
