import os
import os.path as osp
import six
import lmdb
import pickle
import msgpack
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, lengeth, transform=None, target_transform=None):
        self.db_path = db_path
        self.length = lengeth
        self.transform = transform
        self.target_transform = target_transform

    def open_lmdb(self):
         self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                              readonly=True, lock=False,
                              readahead=False, meminit=False)
         self.txn = self.env.begin(write=False, buffers=True)
         self.length = pickle.loads(self.txn.get(b'__len__'))
         self.keys = pickle.loads(self.txn.get(b'__keys__'))

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        
        img, target = None, None
        byteflow = self.txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf[0])
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class ImageFolderLMDB_old(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.keys = msgpack.loads(txn.get(b'__keys__'))
        # cache_file = '_cache_' + db_path.replace('/', '_')
        # if os.path.isfile(cache_file):
        #     self.keys = pickle.load(open(cache_file, "rb"))
        # else:
        #     with self.env.begin(write=False) as txn:
        #         self.keys = [key for key, _ in txn.cursor()]
        #     pickle.dump(self.keys, open(cache_file, "wb"))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = msgpack.loads(byteflow)
        imgbuf = unpacked[0][b'data']
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data

def dumps_pickle(obj):
    """
    Serialize an object.
    
    Returns :
        The pickled representation of the object obj as a bytes object
    """
    return pickle.dumps(obj)

def folder2lmdb(dpath, name="train_images", write_frequency=5000, num_workers=0):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=num_workers)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    map_size = 30737418240 # this should be adjusted based on OS/db size
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=map_size, readonly=False,
                   meminit=False, map_async=True)
    
    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, (data, label) in enumerate(data_loader):
        # print(type(data), data)
        image = data
        label = label.numpy()
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pickle((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pickle(keys))
        txn.put(b'__len__', dumps_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=0)

    args = parser.parse_args()

    folder2lmdb(args.folder, num_workers=args.procs, name=args.split)
