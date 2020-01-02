## LMDB 
LMDB is a json-like, but in binary stream key-value storage. In my design, the format of converted LMDB is defined as follow.

key | value 
--- | ---
img-id1 | (jpeg_raw1, label1)
img-id2 | (jpeg_raw2, label2)
img-id3 | (jpeg_raw3, label3)
... | ...
img-idn | (jpeg_rawn, labeln)
`__keys__` | [img-id1, img-id2, ... img-idn]
`__len__` | n

As for details of reading/writing, please refer to [code](folder2lmdb.py).


## Convert `ImageFolder` to `LMDB`
```bash
python folder2lmdb.py -f ~/torch_data/ -s train
```

OR

You can download the pre-processed lmdb on academic torrents: 
[train.lmdb](https://academictorrents.com/details/d58437a61c1adf9801df99c6a82960d076cb7312),
[val.ldmb](https://academictorrents.com/details/207ebd69f80a3707f035cd91a114466a270e044d).

## ImageFolderLMDB
The usage of `ImageFolderLMDB` is identical to `torchvision.datasets`. 

```python
import ImageFolderLMDB
from torch.utils.data import DataLoader
dst = ImageFolderLMDB(path, transform, target_transform)
loader = DataLoader(dst, batch_size=64)
```

