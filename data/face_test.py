import os
from data import srdata


class face_test(srdata.SRData):
    def __init__(self, args, name='face_test', train=True, benchmark=False):
        super(face_test, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(face_test, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.ext = ('.jpg', '.jpg')
