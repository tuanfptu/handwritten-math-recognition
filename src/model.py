import torch
import torch.nn as nn
import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import hybrid
import gc_module  # Rename the local gc.py to avoid conflict with built-in gc
import transformer


class Model(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def data_parallel(self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)
        if output_device is None:
            output_device = device_ids[0]
        replicas = nn.parallel.replicate(self, device_ids)
        inputs = nn.parallel.scatter(x, device_ids)  # Slices tensors into approximately equal chunks and distributes them across given GPUs.
        kwargs = nn.parallel.scatter(kwargs, device_ids)  # Duplicates references to objects that are not tensors.
        replicas = replicas[:len(inputs)]
        kwargs = kwargs[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)
        return nn.parallel.gather(outputs, output_device).mean()

    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor,  return_logits: bool = False, **kwargs):
        encoded = self.encoder(x)
        # Some decoder configurations (x_transformers.Decoder) expect cross-attention
        # to be enabled/disabled consistently. If decoder was created without
        # cross_attend, passing `context` raises an assertion inside x_transformers.
        # Try calling with context and fall back to calling without it.
        # Determine whether decoder was created with cross-attention enabled.
        cross_attend = False
        try:
            # Common locations depending on wrapper structure
            net = getattr(self.decoder, 'net', None)
            if net is not None and hasattr(net, 'attn_layers'):
                cross_attend = bool(getattr(net.attn_layers, 'cross_attend', False))
            else:
                cross_attend = bool(getattr(self.decoder, 'cross_attend', False))
        except Exception:
            cross_attend = False

        # Immediate diagnostics (stdout) to help trace unexpected decoder outputs
        # try:
        #     net = getattr(self.decoder, 'net', None)
        #     print('---DECODER DEBUG---')
        #     print('decoder_type=', type(self.decoder))
        #     print('has_net=', net is not None)
        #     if net is not None:
        #         print('net_type=', type(net))
        #         print('net_attrs=', [a for a in dir(net) if not a.startswith('_')][:50])
        #         attn_layers = getattr(net, 'attn_layers', None)
        #         print('attn_layers_type=', type(attn_layers))
        #         print('attn_layers_attrs=', [a for a in dir(attn_layers) if not a.startswith('_')][:50])
        #     print('cross_attend=', cross_attend)
        #     print('encoded.shape=', getattr(encoded, 'shape', None))
        #     print('tgt_seq.shape=', getattr(tgt_seq, 'shape', None))
        #     print('kwargs keys=', list(kwargs.keys()))
        #     print('---END DECODER DEBUG---')
        # except Exception as e:
        #     print('debug print failed:', e)

        # If caller requested logits, attempt to extract logits from the
        # AutoregressiveWrapper/TransformerWrapper. Otherwise call decoder as usual.
        out = None
        if return_logits and hasattr(self.decoder, 'net'):
            # try wrapper API first
            try:
                res = self.decoder(tgt_seq, context=encoded if cross_attend else None, return_outputs=True, **kwargs)
                if isinstance(res, tuple) and len(res) == 2:
                    _, tup = res
                    if isinstance(tup, (tuple, list)) and len(tup) >= 1:
                        out = tup[0]
                    else:
                        out = tup
                else:
                    out = res
            except Exception:
                # fallback to calling underlying net directly
                try:
                    net_res = self.decoder.net(tgt_seq, context=encoded if cross_attend else None, return_intermediates=True, **kwargs)
                    if isinstance(net_res, (tuple, list)):
                        out = net_res[0]
                    else:
                        out = net_res
                except Exception:
                    out = None
        if out is None:
            # Normal call (may return loss scalar)
            out = self.decoder(tgt_seq, context=encoded, **kwargs) if cross_attend else self.decoder(tgt_seq, **kwargs)

            # Sanity checks: decoder should produce a 3D tensor (B, T, V)
            if isinstance(out, (tuple, list)):
                out0 = out[0] if len(out) > 0 else None
            else:
                out0 = out

            if out0 is None:
                raise RuntimeError(f'Decoder returned None. encoded.shape={encoded.shape if hasattr(encoded, "shape") else None}, tgt_seq.shape={tgt_seq.shape if hasattr(tgt_seq, "shape") else None}, raw_out={out!r}')

            if not hasattr(out0, 'shape'):
                # Provide as much info as possible about the raw output
                raise RuntimeError(f'Decoder output has no shape attribute. encoded.shape={encoded.shape if hasattr(encoded, "shape") else None}, tgt_seq.shape={tgt_seq.shape if hasattr(tgt_seq, "shape") else None}, raw_out_type={type(out)}, raw_out_repr={repr(out)[:500]}')

            if getattr(out0, 'dim', lambda: None)() != 3:
                # Debug: capture tensor details if it's a tensor-like object
                info = ''
                try:
                    info = f'type={type(out0)}, shape={getattr(out0, "shape", None)}, dtype={getattr(out0, "dtype", None)}, numel={out0.numel() if hasattr(out0, "numel") else None}'
                except Exception:
                    info = f'failed to extract tensor info, raw_out_repr={repr(out)[:500]}'
                raise RuntimeError(f'Unexpected decoder output shape: {getattr(out0, "shape", None)}. encoded.shape={encoded.shape}, tgt_seq.shape={tgt_seq.shape}. DETAILS: {info}')

        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        start = (torch.LongTensor([self.args.bos_token] * len(x))[:, None]).to(x.device)
        ctx = self.encoder(x)
        # Try with context, else without (see forward fallback)
        try:
            return self.decoder.generate(start, self.args.max_seq_len, eos_token=self.args.eos_token, context=ctx, temperature=temperature)
        except AssertionError:
            return self.decoder.generate(start, self.args.max_seq_len, eos_token=self.args.eos_token, temperature=temperature)


def get_model(args):
    if args.encoder_structure.lower() == 'vit':
        encoder = gc_module.get_encoder(args)
    elif args.encoder_structure.lower() == 'hybrid':
        encoder = hybrid.get_encoder(args)
    else:
        raise NotImplementedError('Encoder structure "%s" not supported.' % args.encoder_structure)
    decoder = transformer.get_decoder(args)
    encoder.to(args.device)
    decoder.to(args.device)
    model = Model(encoder, decoder, args)
    if args.wandb:
        import wandb
        wandb.watch(model)
    return model
if __name__ == '__main__':
    from config import get_args
    args = get_args()
    model = get_model(args)
    print(model)