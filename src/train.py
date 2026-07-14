import os
import sys
import logging
from typing import List, Tuple, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
import tqdm
from Dataset import CustomDataset
from model import get_model
from config import get_args


def collate_fn(batch: List[Tuple[dict, torch.Tensor]], pad_token_id: int = 0):
    """Collate function that pads tokenized sequences and stacks images.

    batch: list of tuples (token_dict, image_tensor)
    token_dict contains 'input_ids' and 'attention_mask' tensors (1D each)
    image_tensor is a tensor (C,H,W) or (1,H,W)
    """
    toks = [b[0] for b in batch]
    images = [b[1] for b in batch]

    input_ids = [t['input_ids'].long().squeeze(0) if t['input_ids'].ndim > 1 else t['input_ids'].long() for t in toks]
    attention_masks = [t['attention_mask'].long().squeeze(0) if t['attention_mask'].ndim > 1 else t['attention_mask'].long() for t in toks]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # stack images (expect each image is tensor CxHxW)
    images_stacked = torch.stack(images)

    return {'input_ids': input_ids_padded, 'attention_mask': attention_padded}, images_stacked


def train_epoch(model: nn.Module, dataloader, optimizer, criterion, device: torch.device, scaler: GradScaler = None, accumulate_steps: int = 1):
    model.train()
    total_loss = 0.0
    num_batches = 0
    accum_counter = 0

    pbar = tqdm.tqdm(dataloader, desc='train')
    optimizer.zero_grad()
    for batch in pbar:
        toks, images = batch
        if toks is None or images is None:
            continue

        try:
            images = images.to(device)
            input_ids = toks['input_ids'].to(device)
            attention_mask = toks['attention_mask'].to(device)

            # mixed precision forward
            with autocast(enabled=(scaler is not None and device.type == 'cuda')):
                outputs = model(images, input_ids, return_logits=True)

                # outputs can be either full-length logits (B, seq_len, V) or
                # already-shifted logits (B, seq_len-1, V) depending on decoder wrapper.
                if outputs is None:
                    raise RuntimeError('Decoder returned None')
                if outputs.dim() != 3:
                    raise RuntimeError(f'Unexpected decoder output shape: {outputs.shape} (expected 3 dims)')

                B, L, V = outputs.shape
                input_L = input_ids.shape[1]

                if L == input_L:
                    logits = outputs[:, :-1, :].contiguous()
                    targets = input_ids[:, 1:].contiguous().view(-1)
                elif L == input_L - 1:
                    logits = outputs.contiguous()
                    targets = input_ids[:, 1:].contiguous().view(-1)
                else:
                    raise RuntimeError(f'Unexpected logits length L={L} vs input length {input_L}')

                outputs_flat = logits.view(-1, V)
                loss = criterion(outputs_flat, targets)

            # gradient accumulation: scale loss for backward
            loss_value = loss.item()
            loss_to_backward = loss / accumulate_steps
            if scaler is not None and device.type == 'cuda':
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            accum_counter += 1
            # step optimizer when we've accumulated enough micro-batches
            if accum_counter % accumulate_steps == 0:
                if scaler is not None and device.type == 'cuda':
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except Exception:
                        # if scaler.step fails, try normal step
                        optimizer.step()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                accum_counter = 0

            total_loss += loss_value
            num_batches += 1
            pbar.set_postfix({'loss': total_loss / (num_batches + 1e-12)})

        except RuntimeError as e:
            # catch CUDA OOM and try to recover gracefully
            msg = str(e).lower()
            if 'out of memory' in msg or 'cuda' in msg:
                logging.warning('CUDA out of memory during training step — emptying cache and skipping batch')
                try:
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                except Exception:
                    pass
                continue
            else:
                raise

    return total_loss / max(1, num_batches)


def main(smoke_test: bool = False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    args = get_args()
    device = torch.device(args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu'))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(args.data_root, 'tokenizer.json'),
        unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]'
    )

    df_path = os.path.join(args.data_root, 'metadata.csv')
    df = pd.read_csv(df_path)
    df = df[df['data_source'] == 'CROHME'].reset_index(drop=True)
    df = df[df['tags'] == 'train'].reset_index(drop=True)

    transform = A.Compose([
        # First scale down to fit within max size while preserving aspect ratio
        A.LongestMaxSize(max_size=512),
        # Then pad smaller dimension to reach target size (pad with white=1.0)
        A.PadIfNeeded(min_height=400, min_width=528, border_mode=0, value=1.0),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ])

    dataset = CustomDataset(data=df, tokenizer=tokenizer, max_seq_len=getattr(args, 'max_seq_len', 150))
    dataset.transform = transform

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Prefer using the dataset.get_batch generator if it exists and the user wants it
    use_generator = hasattr(dataset, 'get_batch') and callable(getattr(dataset, 'get_batch'))

    model = get_model(args)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=getattr(args, 'lr', 1e-4))
    criterion = CrossEntropyLoss(ignore_index=pad_token_id)

    # setup mixed precision scaler when using CUDA
    scaler = GradScaler() if device.type == 'cuda' else None

    num_epochs = getattr(args, 'epochs', 10)
    batch_size = getattr(args, 'batch_size', 32)

    # Heuristic: reduce batch size automatically for small GPUs to avoid OOM
    if device.type == 'cuda':
        try:
            props = torch.cuda.get_device_properties(device)
            total_gb = props.total_memory / (1024 ** 3)
            if total_gb < 6:
                # small GPU: prefer micro-batch of 1 with accumulation
                suggested = 1
                default_accum = getattr(args, 'accumulate_steps', 4)
            elif total_gb < 12:
                suggested = min(batch_size, 8)
                default_accum = getattr(args, 'accumulate_steps', 2)
            elif total_gb < 12:
                suggested = min(batch_size, 8)
            else:
                suggested = batch_size
            if suggested != batch_size:
                logging.info(f'Detected GPU with {total_gb:.1f}GB -> adjusting batch_size {batch_size} -> {suggested}')
                batch_size = suggested
        except Exception:
            pass

    if smoke_test:
        num_epochs = 1
        batch_size = 1

    if use_generator:
        # dataset.get_batch yields (tok, images) already batched
        def gen_loader(batch_limit: Optional[int] = None):
            count = 0
            for tok, images in dataset.get_batch(batch_size):
                if tok is None:
                    continue
                yield (tok, images)
                count += 1
                if batch_limit is not None and count >= batch_limit:
                    break

        for epoch in range(num_epochs):
            logging.info(f'Starting epoch {epoch+1}/{num_epochs}')
            batch_limit = 2 if smoke_test else None
            accum_steps = getattr(args, 'accumulate_steps', 4) if batch_size == 1 else getattr(args, 'accumulate_steps', 1)
            avg_loss = train_epoch(model, gen_loader(batch_limit), optimizer, criterion, device, scaler=scaler, accumulate_steps=accum_steps)
            logging.info(f'Epoch {epoch+1} done. avg_loss={avg_loss:.4f}')
            ck = f'model_checkpoint_epoch_{epoch+1}.pt'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, ck)
            logging.info(f'Saved {ck}')
    else:
        # fall back to PyTorch DataLoader with collate_fn
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id))
        for epoch in range(num_epochs):
            logging.info(f'Starting epoch {epoch+1}/{num_epochs}')
            accum_steps = getattr(args, 'accumulate_steps', 4) if batch_size == 1 else getattr(args, 'accumulate_steps', 1)
            avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, scaler=scaler, accumulate_steps=accum_steps)
            logging.info(f'Epoch {epoch+1} done. avg_loss={avg_loss:.4f}')
            ck = f'model_checkpoint_epoch_{epoch+1}.pt'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, ck)
            logging.info(f'Saved {ck}')


if __name__ == '__main__':
    # set smoke_test=True for a short run during debugging
    main(smoke_test=False)
