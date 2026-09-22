# TopoDPO

Reference implementation for **TopoDPO: Dense Preference Alignment for Structure-Sensitive Aerial Referring Segmentation**.

TopoDPO has two parts:

1. **Topo-Prior** mines preference pairs using connected-component discrepancy and macro-corner recall.
2. **Dense preference optimization** applies the DPO nonlinearity to local mask margins before spatial aggregation, with a frozen supervised reference model.

This repository contains the training and evaluation code for the paper. Model weights and image datasets are intentionally not included.

## Repository scope

Included:

- Sa2VA/InternVL model and SAM2 integration code required by the experiments;
- TopoDPO, Vanilla DPO, and TopoDPO-without-SFT configurations;
- Aerial-D and RISORS evaluation scripts;
- the fixed `topo_hard_annotations.json` manifest for Topo-RS-Hard;
- the fixed structural-stress evaluation manifest.

Excluded:

- pretrained or fine-tuned weights;
- Aerial-D/RISORS images and masks;
- optimizer states, work directories, logs, and generated visualizations.

## Environment

The code was developed with Python 3.10, PyTorch, MMEngine, XTuner, Transformers, PEFT, OpenCV, and DeepSpeed. Install the matching Sa2VA/XTuner dependencies for your CUDA and PyTorch environment before running the code.

Install the matching Sa2VA/XTuner dependencies first, then run commands from the repository root with:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

The exact dependency versions may depend on the CUDA/PyTorch installation. Do not install the large model weights through this repository.

## Data and model layout

The original configs expect the following local layout. Edit the paths in `projects/configs/*.py` if your environment differs:

```text
TopoDPO/
├── models/Sa2VA-InternVL3-8B/
├── work_dirs/sft-8b/iter_29056.pth
└── data/Aerial_R1_Dataset/
    ├── sft_mix_train.json
    ├── tdpo_train.json
    ├── aerial-d/
    └── risors/
```

The public benchmark manifest is `topo_hard_annotations.json`. It identifies 4,590 structural-stress instances: 3,721 Aerial-D samples and 869 RISORS samples. The underlying images and masks must be obtained from their respective datasets.

## Training

First train or provide the supervised reference checkpoint expected by the config. Then launch TopoDPO:

```bash
python tools/train.py projects/configs/topo_dpo.py \
  --work-dir work_dirs/topo_dpo
```

Vanilla DPO and the no-SFT ablation are available for comparison:

```bash
python tools/train.py projects/configs/topo_dpo_normal.py \
  --work-dir work_dirs/topo_dpo_normal

python tools/train.py projects/configs/topo_dpo_wo_sft.py \
  --work-dir work_dirs/topo_dpo_wo_sft
```

Use `--resume` to continue from an existing checkpoint. Review all data and model paths before launching a job.

## Checkpoint conversion

Convert a training checkpoint to a Hugging Face-style directory with:

```bash
python tools/convert_to_hf.py projects/configs/topo_dpo.py \
  work_dirs/topo_dpo/iter_8768.pth \
  --save-path models/tdpo-8b
```

The converter writes a new export directory; it does not move or delete the checkpoint.

## Evaluation

Evaluate the two validation sets separately:

```bash
python eval/test_aerial_d.py \
  --data-root /path/to/Aerial_R1_Dataset \
  --json-file /path/to/Aerial_R1_Dataset/aerial-d/val/annotations.json \
  --model models/tdpo-8b \
  --work-dir results/tdpo-8b/aerial_d_test

python eval/test_risors.py \
  --data-root /path/to/Aerial_R1_Dataset \
  --json-file /path/to/Aerial_R1_Dataset/risors/val/annotations.json \
  --model models/tdpo-8b \
  --work-dir results/tdpo-8b/risors_test
```

For the fixed Topo-RS-Hard evaluation, use `eval/test_refseg_hard.py` with the manifest:

```bash
python eval/test_refseg_hard.py \
  --model models/tdpo-8b \
  --dataset both \
  --data-root /path/to/Aerial_R1_Dataset \
  --hard-manifest topo_hard_annotations.json \
  --work-dir results/topo_rs_hard/tdpo-8b
```

The hard-set evaluator reports IoU, F1, precision, recall, Betti errors, topology accuracy, boundary F1, and corner recall. Keep the evaluator version and mask threshold fixed when comparing methods.

## Implementation notes

- The reported Topo-RS-Hard numbers use the fixed 4,590-instance manifest, not a post-hoc selection made from model results.
- Training-time pair mining is offline; connected-component and corner operators are not differentiated through.
- Keep the data split, checkpoint iteration, mask threshold, and evaluation script fixed for fair comparisons.
- The repository does not redistribute third-party datasets or pretrained weights. Check their original licenses before downloading or sharing them.

## Citation

Please cite the paper associated with this repository after the final bibliographic information is available.
