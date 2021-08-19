import os
from data import srdata


class DVI2k(srdata.SRData):
    def __init__(self, args, name='DVI2k', train=True, benchmark=False):
        super(DVI2k, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(DVI2k, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
