# IUDEX

The **<u>I</u>ndiana <u>U</u>niversity <u>D</u>iscourse <u>Ex</u>hibition** (IUDEX) is a collection of parsers and other code related to discourse parsing.

## Setup

For the latest release:

```
pip install larc-iudex
```

For the current state of `master`:

```
pip install git+https://github.com/larc-iu/iudex
```

Or for development:

```
git clone https://github.com/larc-iu/iudex && cd iudex
pip install -e .
```

Note that the command you will invoke is `iudex`, not `larc-iudex`.

## Quick Start with Inference

Parse a sample document end-to-end with a pretrained DMRST model pulled from the HuggingFace Hub. From the command line:

```bash
iudex dmrst predict \
    --hub-id larc-iu/dmrst-gum-12.1.0 \
    --text "Although the experiment was carefully designed, the results were inconclusive. We plan to repeat it tonight."
```
This yields the parsed tree in `.rs3` format printed to `stdout`:
```xml
<rst>
  <relations><!-- ... --></relations>
  <body>
    <segment id="1" parent="2" relname="adversative-concession">Although the experiment was carefully # designed,</segment>
    <segment id="2" parent="4" relname="span">the results were inconclusive.</segment>
    <segment id="3" parent="5" relname="span">We plan to repeat it tonight.</segment>
    <group id="4" type="span" parent="3" relname="adversative-antithesis"/>
    <group id="5" type="span"/>
  </body>
</rst>
```

The same flow from Python:

```python
from iudex.rst.parsers.dmrst.modeling_dmrst import DMRSTParser
parser = DMRSTParser.from_pretrained("larc-iu/dmrst-gum-12.1.0")
tree = parser.predict_from_text(
    "Although the experiment was carefully designed, "
    "the results were inconclusive. "
    "We plan to repeat it tonight."
)
print(tree.to_rs4_string())
```
Yields:
```xml
<rst>
  <relations><!-- ... --></relations>
  <body>
    <segment id="1" parent="2" relname="adversative-concession">Although the experiment was carefully # designed,</segment>
    <segment id="2" parent="4" relname="span">the results were inconclusive.</segment>
    <segment id="3" parent="5" relname="span">We plan to repeat it tonight.</segment>
    <group id="4" type="span" parent="3" relname="adversative-antithesis"/>
    <group id="5" type="span"/>
  </body>
</rst>
```

## Inference CLI

To identify a model on the command line, you may use a configuration file (`--config`), a PyTorch checkpoint (`--checkpoint`), or a HuggingFace Hub repository (`--hub-id`).

To provide input, you may specify an inline string (`--text`), a path to a raw text file or directory (`--text-file`, for parsers which support this), or an RS3/RS4 file or directory with gold EDUs already supplied (`--input`).

For `--text-file` and `--input`, results are written to `--output-dir` as `.rs4` files.

```
# From the Hub, end-to-end on a directory of .txt files:
iudex dmrst predict \
    --hub-id larc-iu/dmrst-gum-12.1.0 \
    --text-file path/to/docs/ \
    --output-dir out/ \
    --device cuda

# From an explicit checkpoint:
iudex dmrst predict \
    --checkpoint checkpoints/<run_id>/best_model.pt \
    --text-file path/to/doc.txt \
    --output-dir out/

# From a trained run's config, parsing pre-segmented RS3/RS4 with gold EDUs:
iudex topdown_biaffine predict \
    --config configs/topdown_biaffine_rstdt.jsonnet \
    --input data/rstdt/test \
    --output-dir out/
```

## Available Models
All official IUDEX model releases are [tagged with `iudex` on the HuggingFace Hub](https://huggingface.co/models?other=iudex).

## Training

To train a new top-down biaffine parser on RSTDT:

```
iudex topdown_biaffine train configs/topdown_biaffine_rstdt.jsonnet
```

Note that `configs/topdown_biaffine_rstdt.jsonnet` is a configuration.
You may either edit it directly or copy and modify it in a new location.

### Grabbing Example Configurations

Model configurations required for training are not bundled with the package distributed via PyPI.

To get them you may visit [the associated directory](https://github.com/larc-iu/iudex/tree/master/configs) and download the configurations you're interested in manually.

If you want to grab all of them at once, you can use the command line like so:

**bash / zsh / macOS / Linux:**

```bash
curl -fL https://github.com/larc-iu/iudex/archive/refs/heads/master.tar.gz \
  | tar -xz --strip-components=1 --wildcards '*/configs'
```

**Windows PowerShell:**

```powershell
Invoke-WebRequest https://github.com/larc-iu/iudex/archive/refs/heads/master.zip -OutFile iudex.zip
Expand-Archive iudex.zip -DestinationPath .
Move-Item iudex-master/configs configs
Remove-Item -Recurse -Force iudex-master, iudex.zip
```

Either leaves you with a local `configs/` directory you can edit and pass to `iudex … train configs/<name>.jsonnet`.

### Configuration Hashes

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
│ 245b1d774676 │ -        │ dmrst            │ xlm-roberta-base             │ data/gum_12.1.0/train │    0.3099 │ 1704 │ 2026-05-18 18:02 │
│ 41bc0fe1dd50 │ -        │ topdown_biaffine │ SpanBERT/spanbert-base-cased │ data/rstdt/train      │    0.7576 │ 2149 │ 2026-05-18 13:51 │
│ 91525e48d63d │ -        │ topdown_biaffine │ SpanBERT/spanbert-base-cased │ data/gum_12.1.0/train │    0.6364 │ 1899 │ 2026-05-18 14:31 │
│ ad934ca992d4 │ -        │ dmrst            │ xlm-roberta-base             │ data/rstdt/train      │    0.4665 │ 3090 │ 2026-05-18 16:46 │
└──────────────┴──────────┴──────────────────┴──────────────────────────────┴───────────────────────┴───────────┴──────┴──────────────────┘
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
