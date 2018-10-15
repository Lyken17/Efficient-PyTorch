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
python folder2lmdb.py --folder ~/torch_data/ --name train
```

## ImageFolderLMDB
The usage of `ImageFolderLMDB` is identical to `torchvision.datasets`. 

```python
import ImageFolderLMDB
from torch.utils.data import DataLoader
dst = ImageFolderLMDB(path, transform, target_transform)
loader = DataLoader(dst, batch_size=64)
```

