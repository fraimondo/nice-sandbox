from mne.externals.h5io import read_hdf5
from mne.io.meas_info import Info

from nice.markers.base import BaseMarker


class BaseMarkerSandbox(BaseMarker):

    def _get_title(self):
        return _get_title(self.__class__, self.comment)


def _get_title(klass, comment):
    if issubclass(klass, BaseMarker):
        kind = 'marker'
    else:
        raise NotImplementedError('Oh no-- what is this?')
    return '/'.join([
        'nice_sandbox', kind, klass.__name__, comment])


def _read_container(klass, fname, comment='default'):
    data = read_hdf5(fname,  _get_title(klass, comment), slash='replace')
    init_params = {k: v for k, v in data.items() if not k.endswith('_')}
    attrs = {k: v for k, v in data.items() if k.endswith('_')}
    file_info = read_hdf5(fname, title='nice/data/ch_info', slash='replace')
    if 'filename' in file_info:
        del(file_info['filename'])
    attrs['ch_info_'] = Info(file_info)
    out = klass(**init_params)
    for k, v in attrs.items():
        if k.endswith('_'):
            setattr(out, k, v)
    return out
