# IUDEX

The ***I*ndiana *U*niversity *D*iscourse *EX*hibition** (IUDEX) is a collection of parsers and other code related to discourse parsing.

## Setup

```
pip install git+https://github.com/larc-iu/iudex
```

or for development:

```
pip install -e .
```

(PyPI deployment will come eventually.)

## Quick Start

### Training
To train a new top-down biaffine parser on RSTDT:

```
iudex topdown_biaffine train configs/topdown_biaffine_rstdt.jsonnet
```

Note that `configs/topdown_biaffine_rstdt.jsonnet` is a configuration.
You may either edit it directly or copy and modify it in a new location.

## Configuration Hashes

Your configuration is used as the basis for a unique hash, which (by default) corresponds to a directory under `checkpoints/`.
This hash is used for several purposes.
For example, running the same config again resumes from the last epoch's checkpoint `last.pt` automatically if the run was interrupted.

To view all runs and their status, you may run the `runs list` subcommand:

```
$ iudex runs list
                                                            Runs in checkpoints                                                            
┏━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ run_id       ┃ run_name ┃ parser           ┃ model_name                   ┃ train_dir             ┃  best_val ┃ step ┃ modified         ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ 007f69457a0c │ -        │ dmrst            │ xlm-roberta-base             │ data/rstdt/train      │ (no best) │    - │ 2026-05-13 16:49 │
│ f4f2bbc875b6 │ -        │ dmrst            │ xlm-roberta-base             │ data/rstdt/train      │ (no best) │    - │ 2026-05-13 16:26 │
└──────────────┴──────────┴──────────────────┴──────────────────────────────┴───────────────────────┴───────────┴──────┴──────────────────┘
```

### Inference

To identify a model, you may use a configuration file (`--config`), a PyTorch checkpoint (`--checkpoint`), or a HuggingFace Hub repository (`--hub-id`).

To provide input, you may specify either an RS3/RS4 input file (`--input`) with gold EDUs already supplied, or (for parsers which support this) a plain text file (`--input-text`).
Both arguments also support directories containing files of the appropriate type.
Examples:

```
# From a trained run, parsing pre-segmented RS3/RS4:
iudex topdown_biaffine predict \
    --config configs/topdown_biaffine_rstdt.jsonnet \
    --input data/rstdt/test \
    --output-dir out/

# From an explicit checkpoint, end-to-end on raw text:
iudex topdown_biaffine predict \
    --checkpoint checkpoints/<run_id>/best_model.pt \
    --input-text path/to/doc.txt \
    --output-dir out/

# From the Hub:
iudex topdown_biaffine predict \
    --hub-id larc-iu/topdown_biaffine-rstdt-coarse \
    --input-text path/to/doc.txt \
    --output-dir out/ \
    --device cuda
```

### Pushing Models to HF Hub
You may host a trained model using each parser's `push` subcommand. 
Each uploads `best_model.pt`, `config.json`, and an auto-generated `README.md` in a single commit:

```
iudex topdown_biaffine push \
    --config configs/topdown_biaffine_rstdt.jsonnet \
    --repo-id larc-iu/topdown_biaffine-rstdt-coarse \
    [--private] [--message "..."] [--token $HF_TOKEN]
```

## Programmatic API

Beyond the CLI, you may also use IUDEX as a library:

```python
from iudex.rst.parsers.topdown_biaffine import TopdownBiaffineParser

parser = TopdownBiaffineParser.from_pretrained("larc-iu/topdown_biaffine-rstdt-coarse")
tree = parser.predict_from_text("Your document text here. Multiple sentences are fine.")
print(tree.to_rs4_string())
```

`from_pretrained` accepts a Hub repo id, a local run directory, or a `.pt` path.
Optional kwargs include `device`, `revision`, `cache_dir`, `token`.

## Citation

If you use IUDEX in your research, please cite it as:

> Gessler, Luke. 2026. *IUDEX: The Indiana University Discourse Exhibition.* https://github.com/larc-iu/iudex.

BibTeX:

```bibtex
@misc{gessler-iudex-2026,
  author       = {Gessler, Luke},
  title        = {{IUDEX: The Indiana University Discourse Exhibition}},
  year         = {2026},
  howpublished = {\url{https://github.com/larc-iu/iudex}},
}
```

If you use one of the included parser re-implementations, please **also** cite the original paper (see each model's Hub card for the canonical reference).
