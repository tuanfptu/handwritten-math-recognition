import os
import logging
from typing import List
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from difflib import SequenceMatcher

from Dataset import CustomDataset
from model import get_model
from config import get_args


def decode_tokens(tokenizer: PreTrainedTokenizerFast, token_ids: List[int]) -> str:
    # remove padding and special tokens
    toks = [t for t in token_ids if t != tokenizer.pad_token_id]
    return tokenizer.decode(toks, skip_special_tokens=True)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def evaluate(model, dataset: CustomDataset, tokenizer: PreTrainedTokenizerFast, device: torch.device, ckpt_path: str = None, batch_size: int = 8, args=None):
    model = model.to(device)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        logging.info(f'Loaded checkpoint {ckpt_path}')

    model.eval()
    # if dataset has no transform, attach a minimal default one
    if getattr(dataset, 'transform', None) is None:
        def _default_transform(image):
            # image: HxWxC (numpy uint8) or HxW
            try:
                arr = np.array(image)
                # scale to fit within max size while preserving aspect ratio
                h, w = arr.shape[:2]
                scale = min(256 / w, 64 / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # pad to reach target size (pad with white=255)
                if arr.ndim == 2:
                    pad_h = max(0, 64 - new_h)
                    pad_w = max(0, 256 - new_w)
                    arr = np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=255)
                    # normalize and add channel dim
                    arr = (arr.astype('float32') / 255.0)[np.newaxis, ...]
                elif arr.ndim == 3:
                    pad_h = max(0, 64 - new_h)
                    pad_w = max(0, 256 - new_w)
                    arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)
                    # normalize and reorder channels
                    arr = (arr.astype('float32') / 255.0).transpose(2, 0, 1)
                
                t = torch.from_numpy(arr)
                return {'image': t}
            except Exception:
                return {'image': torch.tensor([])}

        dataset.transform = _default_transform

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: dataset.get_batch(batch_size) if hasattr(dataset, 'get_batch') else None)

    exact = 0
    total = 0
    sim_sum = 0.0

    # If dataset provides get_batch generator, use it for images/token pairs
    if hasattr(dataset, 'get_batch'):
        gen = dataset.get_batch(batch_size)
        for toks, images in tqdm(gen, desc='Evaluating'):
            if toks is None:
                continue
            # images may be a list/ndarray or a torch.Tensor
            if isinstance(images, list) or isinstance(images, np.ndarray):
                try:
                    images = torch.stack([torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x for x in images])
                except Exception:
                    # images may already be a single tensor from dataset
                    images = torch.as_tensor(images)
            # ensure tensor is float and has batch dim
            if isinstance(images, torch.Tensor):
                if images.dim() == 3:
                    images = images.unsqueeze(1).float()
                else:
                    images = images.float()
                # skip empty image batches (Dataset may yield empty or failed images)
                if not isinstance(images, torch.Tensor) or images.numel() == 0 or images.shape[0] == 0:
                    logging.warning('Skipping empty image batch')
                    continue
            # pad images so H and W are divisible by the encoder patch size
            try:
                b, c, h, w = images.shape
                # get patch size from args or model.encoder
                ps = None
                if args is not None and hasattr(args, 'patch_size'):
                    ps = args.patch_size
                else:
                    enc = getattr(model, 'encoder', None)
                    ps = getattr(enc, 'patch_size', 16) if enc is not None else 16
                # compute padding to next multiple
                pad_h = (-(h) % ps) if (h % ps) != 0 else 0
                pad_w = (-(w) % ps) if (w % ps) != 0 else 0
                if pad_h != 0 or pad_w != 0:
                    # F.pad expects (left, right, top, bottom)
                    images = F.pad(images, (0, pad_w, 0, pad_h), value=1.0)
                    b, c, h, w = images.shape
                # check against model max dimensions (patch grid limits)
                max_h = None
                max_w = None
                if args is not None and hasattr(args, 'max_height') and hasattr(args, 'max_width'):
                    max_h = args.max_height
                    max_w = args.max_width
                else:
                    enc = getattr(model, 'encoder', None)
                    if enc is not None:
                        max_h = getattr(enc, 'max_height', None)
                        max_w = getattr(enc, 'max_width', None)
                if max_h is not None and max_w is not None:
                    if h > max_h or w > max_w:
                        logging.warning(f'Image batch ({h}x{w}) exceeds model max ({max_h}x{max_w}), skipping')
                        continue
            except Exception:
                logging.exception('Error while padding images; skipping batch')
                continue
            images = images.to(device)
            # generate predictions (use model.generate if available)
            with torch.no_grad():
                preds = model.generate(images)

            # preds: tensor BxL
            for i in range(preds.shape[0]):
                pred = decode_tokens(tokenizer, preds[i].tolist())
                target = tokenizer.decode(toks['input_ids'][i].tolist(), skip_special_tokens=True)
                total += 1
                if pred.strip() == target.strip():
                    exact += 1
                sim_sum += similarity(pred, target)
    else:
        # fallback: iterate dataset items
        for i in tqdm(range(len(dataset)), desc='Evaluating'):
            item = dataset[i]
            # item format: dict with 'input_ids' and 'attention_mask' and maybe image path
            # attempt to find corresponding image by index
            try:
                images = item.get('image').unsqueeze(0).to(device)
            except Exception:
                continue
            with torch.no_grad():
                preds = model.generate(images)
            pred = decode_tokens(tokenizer, preds[0].tolist())
            target = tokenizer.decode(item['input_ids'].tolist(), skip_special_tokens=True)
            total += 1
            if pred.strip() == target.strip():
                exact += 1
            sim_sum += similarity(pred, target)

    avg_sim = sim_sum / total if total > 0 else 0.0
    em = exact / total if total > 0 else 0.0
    logging.info(f'Evaluation finished: total={total}, EM={em:.4f}, avg_sim={avg_sim:.4f}')
    return {'total': total, 'EM': em, 'avg_sim': avg_sim}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    device = torch.device(args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.data_root, 'tokenizer.json'), unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
    df_path = os.path.join(args.data_root, 'metadata.csv')
    df = pd.read_csv(df_path)
    df = df[df['data_source'] == 'CROHME'].reset_index(drop=True)
    df = df[df['tags'] == 'test'].reset_index(drop=True)
    dataset = CustomDataset(data=df, tokenizer=tokenizer, max_seq_len=getattr(args, 'max_seq_len', 150), test=True)
    model = get_model(args)
    evaluate(model, dataset, tokenizer, device, ckpt_path=None, batch_size=8)
