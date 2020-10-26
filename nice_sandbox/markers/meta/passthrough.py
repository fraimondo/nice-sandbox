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

from pathlib import Path
import h5py
from mne.utils import logger
from mne.externals.h5io import write_hdf5

from ...markers.base import BaseMarkerSandbox, _read_container


class Passthrough(BaseMarkerSandbox):
    def __init__(self, parent=None, comment='default'):
        BaseMarkerSandbox.__init__(
            self, tmin=None, tmax=None, comment=comment)
        self.parent = parent
        self._check()

    @property
    def _axis_map(self):
        return self.parent._axis_map

    def _check(self):
        # Nothing to check for the moment
        pass

    def _fit(self, epochs):
        if self.parent is None:
            raise ValueError('Need a parent to be able to fit')
        self._check()

        if not hasattr(self.parent, 'data_'):
            logger.warning(
                'Parent not fit. If this is part of a feature collection, '
                'it should be placed after the corresponding estimator.')
            self.parent.fit(epochs)

    @property
    def data_(self):
        logger.warning('This attribute should not be accessed directly '
                       'as it depends on the parent data_ '
                       'attribute')
        return self.parent.data_

    def save(self, fname, overwrite=False):
        if not isinstance(fname, Path):
            fname = Path(fname)
        self._save_info(fname, overwrite=overwrite)
        save_vars = self._get_save_vars(
            exclude=['data_', 'parent', 'tmin', 'tmax'])

        parent_name = self.parent._get_title()

        save_vars['parent_name_'] = parent_name

        has_parent = False

        with h5py.File(fname) as h5fid:
            if parent_name in h5fid:
                has_parent = True
                logger.info('Parent already present in HDF5 file, '
                            'will not be overwritten')

        if not has_parent:
            logger.info('Writing numerator to HDF5 file')
            self.parent.save(fname, overwrite=overwrite)

        write_hdf5(
            fname, save_vars, overwrite=overwrite,
            title=self._get_title(), slash='replace')

    @classmethod
    def _read(cls, fname, markers=None, comment='default'):
        return _read_passthrough(
            cls, fname=fname, markers=markers, comment=comment)


def _read_passthrough(cls, fname, markers=None, comment='default'):
    out = _read_container(cls, fname, comment=comment)
    if markers is None:
        # Get class of each estimator and read
        raise NotImplementedError(
            'This feature is not yet implemented. Please read all the '
            'markers first')

    out.parent = markers[out.parent_name_]
    del out.parent_name_
    return out


def read_passthrough(fname, markers=None, comment='default'):
    out = Passthrough._read(fname, markers=markers, comment=comment)
    return out
