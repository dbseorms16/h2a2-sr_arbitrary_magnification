import os
from data import srdata


class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )
        self.total_scale = args.total_scale

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, 'benchmark', 'face_test')
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.ext = ('.jpg', '.jpg')

