# Handwritten Math Recognition

End-to-end recognition of handwritten mathematical expressions: image input to LaTeX and label-graph output. The project combines a ViT encoder + Transformer decoder with a Qwen3-VL LoRA inference service, and includes the CROHME 2019 data, training notebooks, labeling tools, web clients, and a recorded demo.

## Demo

https://github.com/user-attachments/assets/4821be03-e62f-48c6-9597-fb6648b1f6e7

## Highlights

- ViT/Hybrid visual encoder and autoregressive Transformer decoder
- Qwen3-VL LoRA API for image-to-LaTeX inference
- CROHME 2019 train/validation/test samples and metadata
- EDA, preprocessing, and training notebooks
- Browser, Streamlit, and Gradio interfaces
- Docker Compose deployment with NVIDIA GPU support
- Label-graph conversion and desktop data-labeling utilities

## Architecture

```text
handwritten image
      |
      +--> ViT / Hybrid Encoder --> Transformer Decoder --> LaTeX
      |
      +--> Qwen3-VL + LoRA -------> API -------> Web / Gradio / Streamlit
```

## Repository layout

```text
.
|-- apps/                 # Inference API, Docker, and UI clients
|   `-- web/              # Browser, Streamlit, and Gradio clients
|-- assets/               # Demo video and sample predictions
|-- data/                 # CROHME 2019, metadata, tokenizer
|-- docs/                 # Project report and experiment figures
|-- models/               # Local checkpoints (Git-ignored)
|-- notebooks/            # EDA, preprocessing, and training
|-- src/                  # Dataset, model, training, evaluation
|-- tools/                # Synthetic generation and labeling tools
|-- requirements.txt      # Core training dependencies
`-- requirements-app.txt  # API/UI dependencies
```

## Quick start

Python 3.10+ and a CUDA-capable GPU are recommended.

```bash
git clone <repository-url>
cd handwritten-math-recognition
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Evaluate a trained ViT checkpoint with:

```bash
python src/evaluate.py
```

The default paths are resolved relative to the repository (`data/` and `models/`) in `src/config.py`.

## Models

Pretrained weights are not committed because each artifact exceeds GitHub's 100 MB per-file limit. Copy these original project artifacts into `models/`:

| Artifact | Purpose | Approx. size |
|---|---|---:|
| `vit_seq2seq_.pt` | ViT + Transformer checkpoint | 297 MB |
| `lora_model_qwen3vl.zip` | Qwen3-VL LoRA adapter | 165 MB |

See [`models/README.md`](models/README.md). For distribution, attach them to a GitHub Release or publish them on Hugging Face.

## Run the Qwen3-VL service

Install application dependencies and set the local model path:

```powershell
pip install -r requirements-app.txt
$env:MODEL_PATH = (Resolve-Path models\lora_model_qwen3vl.zip)
python apps/main.py
```

Or use Docker with an NVIDIA Container Toolkit-enabled host:

```bash
cd apps
docker compose up --build
```

The API listens on `http://localhost:8080`. Start a client in another terminal:

```bash
streamlit run apps/web/app.py
# or
python apps/web/app_gradio_dual.py
```

## Data and notebooks

`data/` contains the CROHME 2019 splits, `metadata.csv`, and `tokenizer.json`. The research workflow is documented in:

1. `notebooks/EDA.ipynb`
2. `notebooks/DataProcess.ipynb`
3. `notebooks/train_vit_crohme.ipynb`

The full academic report is available at [`docs/report.pdf`](docs/report.pdf).

## Demo assets

The original high-resolution recording is preserved at [`assets/demo.mp4`](assets/demo.mp4). Sample inputs and experiment outputs are available in `assets/` and `docs/`.

## Reproducibility notes

- Large model weights are intentionally ignored by Git.
- Generated outputs, local virtual environments, caches, and secrets are ignored.
- The original research notebooks are preserved; command-line scripts use repository-relative paths.
- GPU memory requirements vary by backend. Qwen3-VL inference is the most demanding path.

## Acknowledgements

Built as a research project at FPT University using CROHME, PyTorch, Hugging Face Transformers, timm, LitServe, Streamlit, and Gradio.

## License

No open-source license has been selected yet. Copyright remains with the project authors.
