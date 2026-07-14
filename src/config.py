from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_args():
    return ModelConfig()

class ModelConfig:
    encoder_structure: str = 'vit'  # Options: 'vit', 'hybrid'
    decoder_structure: str = 'transformer'  # Currently only 'transformer' is supported
    device: str = 'cuda'  # Use 'cpu' on machines without CUDA.
    data_root: str = str(PROJECT_ROOT / 'data')
    model_root: str = str(PROJECT_ROOT / 'models')
    num_tokens: int = 512  # Vocabulary size
    max_seq_len: int = 256  # Maximum sequence length for the decoder
    dim: int = 512  # Dimension of the model
    num_layers: int = 6  # Number of layers in the transformer
    num_heads: int = 6  # Number of attention heads
    dim_head: int = 32  # Dimension of each attention head
    dim_ff: int = 2048  # Dimension of the feedforward network
    dropout: float = 0.1  # Dropout rate
    attn_dropout: float = 0.1  # Attention dropout rate
    ff_dropout: float = 0.1  # Feedforward dropout rate
    bos_token: int = 1  # Beginning of sequence token ID
    eos_token: int = 2  # End of sequence token ID
    pad_token: int = 0  # Padding token ID
    wandb: bool = False  # Whether to use Weights & Biases for logging
    decoder_args: dict = {}  # Additional arguments for the decoder
    encoder_args: dict = {}  # Additional arguments for the encoder
    hybrid_args: dict = {}  # Additional arguments for the hybrid encoder (if used)
    gc_args: dict = {}  # Additional arguments for the gc encoder (if used)
    channels: int = 1  # Number of input channels (e.g., 1 for grayscale images)
    max_height: int = 400  # Maximum height of input images
    max_width: int = 528  # Maximum width of input images
    patch_size: int = 1  # Patch size for the ViT encoder
    emb_dropout: float = 0.1  # Embedding dropout rate for the encoder
    encoder_depth: int = 4  # Depth of the encoder (number of layers)
    heads: int = 4  # Number of attention heads in the encoder
    dim_head: int = 32  # Dimension of each attention head in the encoder
    ff_dropout: float = 0.1  # Feedforward dropout rate in the encoder
    ff_mult: int = 4  # Feedforward network multiplier in the encoder
    patch_dropout: float = 0.1  # Dropout rate for patches in the encoder
    gc_args: dict = None  # Additional arguments for the gc encoder (if used)
