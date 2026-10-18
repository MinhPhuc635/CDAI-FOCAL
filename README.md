# FOCAL with a DINOv3 Backbone

Fork of [tomoyoshki/focal](https://github.com/tomoyoshki/focal) (NeurIPS 2023, [arXiv:2310.20071](https://arxiv.org/abs/2310.20071)).

**What changed:** the encoder (backbone) is replaced by **DINOv3**. The FOCAL losses, data pipeline and training procedure are unchanged. For the paper's method, datasets, preprocessing and the original DeepSense / Swin-Transformer backbones, see the [original repository](https://github.com/tomoyoshki/focal).

## Summary of changes

| File | Change |
|---|---|
| `src/models/DINOv3.py` | **New.** DINOv3 backbone, selected with `-model=DINOv3`. |
| `src/train_utils/model_selection.py` | Imports `DINOv3` and adds the branch `elif args.model == "DINOv3"` in `init_backbone_model`. |
| `src/data/MOD.yaml` | Adds a `DINOv3:` config block and two per-model entries under `FOCAL`; lowers the pretraining learning rate. |

### How the backbone works (`src/models/DINOv3.py`)

For each modality, the FFT input `[b, c, i, s]` goes through:

1. a signed-log transform of the real/imaginary parts (plus magnitude),
2. a learnable 1x1 convolution that maps the channels to 3 (like an RGB image),
3. bilinear resizing to `img_size x img_size`,
4. a DINOv3 encoder (one per modality), taking the CLS token,
5. a linear layer to `loc_out_channels`.

It exposes the same interface as `SW_Transformer` (`forward(freq_x, class_head, proj_head)`), so no change is needed in the FOCAL loss or training code.

### Config changes (`src/data/MOD.yaml`)

- Add a top-level `DINOv3:` block with `pretrained_name`, `img_size` (112), `loc_out_channels` (256), `pretrained_head`, `optimizer`, `lr_scheduler`, `random_augmenters`, `fixed_augmenters`.
- Add `DINOv3: 0.07` under `FOCAL -> temperature` and `DINOv3: 0.005` under `FOCAL -> finetune_optimizer -> weight_decay`. Both are looked up by model name; a missing entry raises `KeyError: 'DINOv3'`.
- Lower `FOCAL -> pretrain_optimizer -> start_lr` from `0.001` to `0.0001` to avoid damaging the pretrained DINOv3 weights.

## Setup

Install dependencies as in the original repository, then add:

```bash
pip install -U "transformers>=4.56" huggingface_hub
```

On Google Colab, do **not** run `pip install -r requirements.txt` (it pins an old PyTorch/CUDA stack). Install only what is needed:

```bash
pip install -q "transformers>=4.56" tsai timm einops scikit-learn pyyaml tqdm huggingface_hub
```

### DINOv3 weights (gated model)

1. Create a Hugging Face account and accept the license on the model page, e.g. `facebook/dinov3-vits16-pretrain-lvd1689m`.
2. Create a *Read* access token (Settings -> Access Tokens).
3. Log in before running the code:

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")   # on Colab: store it in Secrets as HF_TOKEN
```

If access has not been granted yet, `facebook/dinov2-small` (not gated) can be used as a drop-in substitute by changing `pretrained_name` in `MOD.yaml`.

## Usage (DINOv3)

Run all commands from `src/`. Data preparation and index files are the same as in the original repository.

```bash
# 1. Pretraining
python train.py -model=DINOv3 -dataset=MOD -learn_framework=FOCAL -batch_size=64

# 2. Fine-tuning
python train.py -model=DINOv3 -dataset=MOD -learn_framework=FOCAL \
    -task=vehicle_classification -stage=finetune \
    -model_weight=../weights/MOD_DINOv3/exp0_contrastive_FOCAL

# 3. Testing
python test.py -model=DINOv3 -dataset=MOD -stage=finetune \
    -task=vehicle_classification \
    -model_weight=../weights/MOD_DINOv3/exp0_contrastive_FOCAL
```

Weights are saved under `weights/MOD_DINOv3/`. The first run creates `exp0_contrastive_FOCAL`, later runs create `exp1_...`, `exp2_...`.

### Notes for a 16 GB GPU (e.g. Colab T4)

- The default batch size of 256 runs out of memory. Use `-batch_size=64` (must be divisible by the sequence length 4). A smaller batch means fewer negatives for contrastive learning, which may slightly reduce quality.
- The default of 6000 pretraining epochs is not practical. Run a 2-epoch smoke test first, then increase `train_epochs` under `FOCAL -> pretrain_lr_scheduler` (e.g. 100).
- If memory is still insufficient, reduce `img_size` in the `DINOv3:` block.

## Limitations

DINOv3 was pretrained on natural RGB images, while the input here is a time-frequency map, so results may differ from the original paper's backbones and should be validated experimentally. DINOv3 weights are distributed by Meta under their own license; read it on the Hugging Face model page before use.

## License and Citation

MIT license (see `LICENSE`). Please cite the original paper:

```bibtex
@inproceedings{liu2023focal,
  title = {FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space},
  author = {Liu, Shengzhong and Kimura, Tomoyoshi and Liu, Dongxin and Wang, Ruijie and Li, Jinyang and Diggavi, Suhas and Srivastava, Mani and Abdelzaher, Tarek},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023}
}
```
