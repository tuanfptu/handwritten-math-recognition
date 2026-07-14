from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import cv2
import imagesize
import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import logging
import torch.nn.functional as F
from pathlib import Path
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer: PreTrainedTokenizerFast, max_seq_len: int,max_height:int=64, max_width:int=256,test:bool=False, transform=None, data_root=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_height = max_height
        self.max_width = max_width
        self.Data = dict()
        self.pad = False
        self.test = test
        self.data_root = Path(data_root or Path(__file__).resolve().parents[1] / "data")

        try:
            for i, im in tqdm.tqdm(enumerate(data['name']), total=len(data['name'])):
                try:
                    image_path = self.data_root / (im.removesuffix('.inkml') + '.png')
                    width, height = imagesize.get(str(image_path))
                    if (width, height) not in self.Data:
                        self.Data[(width, height)] = []
                    self.Data[(width, height)].append((data['Latex'][i], str(image_path)))
                except FileNotFoundError:
                    pass
        except KeyboardInterrupt:
            pass

    def __len__(self):
        return len(self.data)
    def get_batch(self, batch_size):
        for key in self.Data:
            data = np.array(self.Data[key])
            for i in range(0, len(data), batch_size):
                d = data[i:i + batch_size]
                tok = self.tokenizer(d[:, 0].tolist(),return_token_type_ids=False)
                for k, p in zip(tok, [[1,2], [1, 1]]):
                    tok[k] = pad_sequence([torch.LongTensor([p[0]]+x+[p[1]]) for x in tok[k]], batch_first=True, padding_value=0)
                images = []
                for path in list(d[:, 1]):
                    im = cv2.imread(path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    if not self.test:
                        # sometimes convert to bitmask
                        if np.random.random() < .04:
                            im[im != 255] = 0
                    if self.transform is not None:
                        images.append(self.transform(image=im)['image'][:1])
                try:
                    images = torch.cat(images).float().unsqueeze(1)
                except RuntimeError:
                    logging.critical('Images not working: %s' % (' '.join(list(d[:, 1]))))
                    yield None, None
                if self.pad:
                    h, w = images.shape[2:]
                    images = F.pad(images, (0, self.max_dimensions[0]-w, 0, self.max_dimensions[1]-h), value=1)
                yield tok, images
    def __getitem__(self, idx):
        item = self.data[idx]
        image = cv2.imread(item['Latex'], cv2.IMREAD_GRAYSCALE)
        label = item['Latex']
        tokenized_label = self.tokenizer(label, padding='max_length', truncation=True, max_length=self.max_seq_len, return_tensors='pt')
        input_ids = tokenized_label.input_ids.squeeze(0)  # Remove batch dimension
        attention_mask = tokenized_label.attention_mask.squeeze(0)  # Remove batch dimension
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
if __name__ == "__main__":
    from transformers import PreTrainedTokenizerFast
    import pandas as pd
    data_root = Path(__file__).resolve().parents[1] / "data"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(data_root / "tokenizer.json"), unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    df = pd.read_csv(data_root / "metadata.csv")
    df = df[df["data_source"] == "CROHME"].reset_index()
    df = df[df["tags"] == "train"].reset_index()
    dataset = CustomDataset(df, tokenizer, max_seq_len=150, data_root=data_root)
