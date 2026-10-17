# FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space

Authors: **Shengzhong Liu, Tomoyoshi Kimura, Dongxin Liu, Ruijie Wang, Jinyang Li, Suhas Diggavi, Mani Srivastava, Tarek Abdelzaher**

Arxiv Link [[pdf](https://arxiv.org/abs/2310.20071)]

## Overview

This paper proposes a novel contrastive learning framework, called FOCAL, for extracting comprehensive features from multimodal time-series sensing signals through self-supervised training. Existing multimodal contrastive frameworks mostly rely on the shared information between sensory modalities, but do not explicitly consider the exclusive modality information that could be critical to understanding the underlying sensing physics. Besides, contrastive frameworks for time series have not handled the temporal information locality appropriately. FOCAL solves these challenges by making the following contributions: First, given multimodal time series, it encodes each modality into a factorized latent space consisting of shared features and private features that are orthogonal to each other. The shared space emphasizes feature patterns consistent across sensory modalities through a modal-matching objective. In contrast, the private space extracts modality-exclusive information through a transformation-invariant objective. Second, we propose a temporal structural constraint for modality features, such that the average distance between temporally neighboring samples is no larger than that of temporally distant samples. Extensive evaluations are performed on four multimodal sensing datasets with two backbone encoders and two classifiers to demonstrate the superiority of FOCAL. It consistently outperforms the state-of-the-art baselines in downstream tasks with a clear margin, under different ratios of available labels.

### [NEW] Modifications in this fork

This fork extends the original FOCAL code with the following changes:

- **New backbone:** a **DINOv3** encoder (`-model=DINOv3`, file `src/models/DINOv3.py`) in addition to the original DeepSense and Swin-Transformer (`SW_Transformer`). The FOCAL losses, data pipeline and training procedure are unchanged; only the encoder is replaced.
- **Google Colab workflow:** step-by-step setup for running on Colab, with the dataset downloaded from Kaggle.
- **Dataset download and preprocessing:** a file-by-file download script for the Kaggle dataset, and the full preprocessing order including the pretraining data.
- **Small README fixes** to commands that were incorrect in the original (marked below).

## Installation and Requirements

### [UPDATE] Local machine

1. **Dependencies**: ```conda create -n [NAME] python=3.10; conda activate [NAME]```

2. **Installation**:

```bash
    git clone [repo] [dir]
    cd [dir]
    pip install -r requirements.txt
```

3. [NEW] **For the DINOv3 backbone**, additionally install a recent Transformers library:

```bash
    pip install -U "transformers>=4.56" huggingface_hub
```

### [NEW] Google Colab

Do **not** run `pip install -r requirements.txt` on Colab: it pins an old PyTorch/CUDA stack and conflicts with the Colab environment. Install only what is needed:

```bash
git clone https://github.com/tomoyoshki/focal.git
cd focal/src
pip install -q "transformers>=4.56" tsai timm einops scikit-learn pyyaml tqdm huggingface_hub
```

Select a GPU runtime (**Runtime → Change runtime type → T4 GPU**).

### [NEW] Getting the DINOv3 weights

DINOv3 is a gated model on Hugging Face.

1. Create a Hugging Face account and accept the license on the model page, e.g. `facebook/dinov3-vits16-pretrain-lvd1689m`.
2. Create a *Read* access token (Settings → Access Tokens).
3. Log in before running the code:

```python
    from huggingface_hub import login
    login(token="YOUR_HF_TOKEN")   # on Colab: store it in Secrets as HF_TOKEN
```

If access has not been granted yet, `facebook/dinov2-small` (not gated) can be used as a drop-in substitute by changing `pretrained_name` in the config.

## Dataset

### Moving Object Detection (MOD)

MOD is a self-collected dataset using acoustic (8000Hz) and seismic (100Hz) signals to classify moving vehicle types. The pretrain dataset includes lots of 10 classes, and the downstream tasks include vehicle classification, distance classification, and speed classification.

- Vehicle data: [Box link](https://uofi.box.com/s/bv37vqfd0a5d9rhnvfoo7gld96gtj8jv)
- Distance speed raw data: [Box link](https://uofi.box.com/s/8yffx3417mrxbdsqqtder4d7kk7rryb8)

#### [NEW] Kaggle mirror

The raw vehicle data is also available as a Kaggle dataset: `nguyenmp/focal-dataset` (about 21.7 GB, 579 files, structure `MOD_vehicle/raw_data/<vehicle>/<sensor>/...`).

Downloading the whole dataset at once can fail with a 404 error for large datasets, so download file by file. The preprocessing code only reads `ehz.csv` (seismic) and `aud16000.csv` / `aud.csv` (acoustic); other files (`*.wav`, `gps.csv`, `ene.csv`, `enn.csv`, `enz.csv`, `*.png`, `timestamps.csv`) are not used and can be skipped.

```bash
# Kaggle API token: Kaggle → Settings → API → Generate New Token
export KAGGLE_API_TOKEN=YOUR_KAGGLE_TOKEN
kaggle datasets download -d nguyenmp/focal-dataset \
    -f MOD_vehicle/raw_data/<vehicle>/<sensor>/ehz.csv -p <output_dir> --unzip
```

A script that lists all files and downloads only the required ones (with retries and size checks) is given in the project notes; it can be re-run safely and skips files that are already complete.

### Data preprocessing

1. Update directories in `src/data_preprocess/MOD/preprocessing_config.py` [UPDATE] (the original README wrote `proprocessing_configs.py`, which does not exist):
   - `input_path`: path to `.../MOD_vehicle/raw_data`
   - `output_directory`: where processed samples are written (e.g. `/content/processed` on Colab)
2. `cd src/data_preprocess/MOD/`
3. Extract samples from raw data
4. Partition data
5. Please update the index path configurations in `src/data/*.yaml`

[NEW] **Order matters.** To use the full dataset (labeled data plus the extra vehicles for pretraining), run the four scripts in this exact order, since each uses the output of the previous one:

1. `extract_samples.py`
2. `partition_data.py`
3. `extract_pretrain_samples.py`
4. `partition_data_pretrain.py`

#### Sample Extraction

##### Extract pretrain data

```bash
python extract_pretrain_samples.py
```

##### Extract supervised data

```bash
python extract_samples.py
```

##### Extract speed and distance data

```bash
python extract_samples_speed_distance.py
```

#### Data Partition

##### Partition pretrain data

```bash
python partition_data_pretrain.py
```

##### Partition supervised data

```bash
python partition_data.py
```

##### Partition speed_distance data

[UPDATE] (the original command was missing the `.py` extension)

```bash
python partition_data_speed_distance.py
```

#### [NEW] Index files and paths

After partitioning, `output_directory/time_data_partition/` contains `train_index.txt`, `val_index.txt`, `test_index.txt` and `pretrain_index.txt`. Each line is the absolute path of one `.pt` sample. If you preprocess on one machine and train on another, rewrite the path prefix in these files, e.g.:

```bash
sed -i 's#/old/prefix#/new/prefix#g' time_data_partition/*.txt
```

Then set the four paths in `src/data/MOD.yaml` (`pretrain_index_file`, and `train_index_file`, `val_index_file`, `test_index_file` under `vehicle_classification`).

## Usage

### Training + Fine-tuning

#### Documentation

Please download the dataset from the provided links in the Dataset section. The arguments to run the training program can be found by running

```bash
python train.py --help
```

All commands below must be run from the `src/` directory, because the code reads `./data/[DATASET].yaml` and writes weights to `../weights`. [NEW]

#### Supervised Training

[UPDATE] The following command will run supervised training for DeepSense (`-model=DeepSense`), Transformer (`-model=SW_Transformer`), or DINOv3 (`-model=DINOv3`).

```bash
python train.py -model=[MODEL] -dataset=[DATASET]
```

#### FOCAL Pre-training

```bash
python train.py -model=[MODEL] -dataset=[DATASET] -learn_framework=FOCAL
```

[NEW] **Notes for the DINOv3 backbone on a 16 GB GPU (e.g. Colab T4):**

- The default batch size of 256 runs out of memory. Use `-batch_size=64` (must be divisible by the sequence length 4). Fewer samples per batch means fewer negatives for contrastive learning, which may slightly reduce quality.
- The default of 6000 pretraining epochs is not practical on Colab. Do a smoke test with 2 epochs first, then increase `train_epochs` under `FOCAL → pretrain_lr_scheduler` in the YAML (e.g. 100).
- Example:

```bash
    python train.py -model=DINOv3 -dataset=MOD -learn_framework=FOCAL -batch_size=64
```

#### FOCAL Fine-tuning with specific downstream task

```bash
python train.py -model=[MODEL] -dataset=[DATASET] -learn_framework=FOCAL -task=[TASK] -stage=finetune -model_weight=[PATH TO MODEL WEIGHT]
```

- Note that the path to the model weight is the folder under the generated `weights/[DATASET]_[MODEL]/` directory.
- [NEW] The first pretraining run creates `weights/MOD_DINOv3/exp0_contrastive_FOCAL/`; later runs create `exp1_...`, `exp2_...`, and so on.

### Testing

#### Supervised

[UPDATE] (the original command was missing `=` after `-dataset`)

```bash
python test.py -model=[MODEL] -dataset=[DATASET] -model_weight=[PATH TO MODEL WEIGHT]
```

#### FOCAL

[UPDATE] (same fix)

```bash
python test.py -model=[MODEL] -dataset=[DATASET] -stage=finetune -task=[TASK] -model_weight=[PATH TO MODEL WEIGHT]
```

## Model Configs

See the file `src/data/*.yaml` for specific model configurations for each dataset.

### [NEW] DINOv3 configuration

To add the DINOv3 backbone, the following changes are made:

1. **New file `src/models/DINOv3.py`.** For each modality it applies a signed-log transform to the FFT input, a learnable 1×1 convolution to map the channels to 3, bilinear resizing to `img_size × img_size`, a DINOv3 encoder (one per modality), and takes the CLS token followed by a linear layer to `loc_out_channels`. It exposes the same interface as `SW_Transformer` (`forward(freq_x, class_head, proj_head)`), so the FOCAL loss and training code are unchanged.

2. **`src/train_utils/model_selection.py`:** import `DINOv3` and add a branch `elif args.model == "DINOv3": classifier = DINOv3(args)` in `init_backbone_model`.

3. **`src/data/MOD.yaml`:**
   - Add a new top-level `DINOv3:` block with `pretrained_name`, `img_size` (112), `loc_out_channels` (256), `pretrained_head`, `optimizer`, `lr_scheduler`, `random_augmenters`, `fixed_augmenters`.
   - Add `DINOv3: 0.07` under `FOCAL → temperature`, and `DINOv3: 0.005` under `FOCAL → finetune_optimizer → weight_decay`. Both tables are looked up by model name, so a missing entry causes a `KeyError`.
   - Lower `FOCAL → pretrain_optimizer → start_lr` from `0.001` to `0.0001` to avoid damaging the pretrained DINOv3 weights.

Because the DINOv3 encoder was pretrained on natural RGB images while the input here is a time-frequency map, results may differ from the original paper's backbones and should be validated experimentally.

## License

This project is released under the MIT license. See `LICENSE` for details. [NEW] DINOv3 weights are distributed by Meta under their own license; please read it on the Hugging Face model page before use.

## Citation

```latex
@inproceedings{liu2023focal,
  title = {FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space},
  author = {Liu, Shengzhong and Kimura, Tomoyoshi and Liu, Dongxin and Wang, Ruijie and Li, Jinyang and Diggavi, Suhas and Srivastava, Mani and Abdelzaher, Tarek},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023}
}
```

## Contact

For any issue regarding the paper or the code, please contact us.

- Shengzhong Liu: <shengzhong@sjtu.edu.cn>
- Tomoyoshi Kimura: <tkimura4@illinois.edu>
