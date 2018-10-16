import numpy as np
from tensorpack.dataflow import *
class BinaryILSVRC12(dataset.ILSVRC12Files):
    def __iter__(self):
        for fname, label in super(BinaryILSVRC12, self).__iter__():
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, label]


from tensorpack.dataflow.serialize import LMDBSerializer
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str)
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=20)

    args = parser.parse_args()
    
    import os.path as osp
    ds0 = BinaryILSVRC12(args.ds, args.split)
    ds1 = PrefetchDataZMQ(ds0, nr_proc=args.procs)
    LMDBSerializer.save(ds1,  osp.join(args.out, '%s.lmdb' % args.split))
