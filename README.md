# GRASP - Generic Reasoning and SPARQL generation across Knowledge Graphs

## News

- August 28th 2025:
  - Demo paper of GRASP has also been accepted to [ISWC 2025](https://iswc2025.semanticweb.org/)
  - Preview of camera-ready version coming soon

- August 23rd 2025:
  - Changes:
    - Major refactor
    - New index format and updated indices for all knowledge graphs
    - Additional information for entities / properties is now loaded online
      from a live SPARQL endpoint

  - New CLI as part of the refactor:
    - `grasp run <config> <input>`: Run GRASP on an input
    - `grasp file <config> <file>`: Run GRASP on a file with inputs
    - `grasp serve <config>`: Start a GRASP server
    - `grasp data <kg>`: Download data for a knowledge graph
    - `grasp merge <kg1> <kg2> ... <kg_out>`: Merge data of multiple knowledge graphs
    - `grasp index <kg>`: Build indices for a knowledge graph
    - `grasp evaluate <file> <pred> <endpoint>`: Evaluate GRASP predictions for
    a input file against a SPARQL endpoint

- July 31st 2025:
  - GRASP has been accepted to [ISWC 2025](https://iswc2025.semanticweb.org/)
  - Preview of camera-ready version available [here](https://ad-publications.cs.uni-freiburg.de/ISWC_grasp_WB_2025.pdf)

- July 14th 2025:
  - arXiv preprint available at [arxiv.org/abs/2507.08107](https://arxiv.org/abs/2507.08107)

- July 10th 2025:
  - Code release
  - Data release

## Overview and directory structure

Data available at [ad-publications.cs.uni-freiburg.de/grasp](https://ad-publications.cs.uni-freiburg.de/grasp)

```
Makefile                          # Makefile for building benchmarks
src/                              # Source code for GRASP
bash/                             # Bash scripts to run and evaluate GRASP
scripts/                          # Various helper scripts
app/
  evaluation/                     # Streamlit app for evaluation
data/                          
  benchmark/                      # Benchmarks grouped by knowledge graph
    [knowledge-graph]/
      [benchmark]/                   
        test.jsonl                # Test set with input and ground truth
        train.example_index/      # Index based on train set for few-shot learning
                                    (needs to be downloaded)
        outputs/
          [model].jsonl           # Model output
          [model].config.json     # Model config
          [model].evaluation.json # Evaluation against ground truth
  kg-index/                       # KG indices (need to be downloaded)
    wikidata/
    freebase/
    ...
configs/
  run.yaml                        # Config to run GRASP with a single KG
  serve.yaml                      # Config to run GRASP with all available KGs
```

## Quickstart

Follow these steps to run GRASP and the evaluation app.

### Run GRASP

> Note: We recommend to use conda for ease of installation of Faiss and to avoid
> dependency issues.

1. Create and activate conda environment:
`conda create -n grasp python=3.12 && conda activate grasp`

2. Install Faiss (not supported to be installed with pip):
`conda install -c pytorch -c nvidia faiss-gpu=1.11.0`

> You might have to install the CPU version of Faiss, since
> the GPU version leads to issues on some systems.

3. Clone the repository: `git clone https://github.com/ad-freiburg/grasp`

4. Go to directory and install with pip: `cd grasp && pip install -e .`

5. Set the `GRASP_INDEX_DIR` env variable. Defaults to `$HOME/.grasp/index` if not
set. We set it to `$PWD/data/kg-index`, but you can choose any directory you like.

> We recommend to set it with conda, such that it is set automatically when you activate
> the conda environment: `conda env config vars set GRASP_INDEX_DIR=/path/to/dir`

6. Get indices for the knowledge graphs you want to use. All indices are available
[publicly](https://ad-publications.cs.uni-freiburg.de/grasp/kg-index).
For example, to get the indices for Wikidata:

```bash
# change to index directory
cd $GRASP_INDEX_DIR
# download Wikidata index
wget https://ad-publications.cs.uni-freiburg.de/grasp/kg-index/wikidata.tar.gz
# extract index
tar -xzf wikidata.tar.gz
```

Optionally, you can also download example indices for few-shot learning.
Example indices are always built from the train set of a benchmark
and called `train.example-index`.
For example, to get the example index for QALD-10 on Wikidata:

```bash
# change to benchmark directory
cd data/benchmark/wikidata/qald10
# download example index
wget https://ad-publications.cs.uni-freiburg.de/grasp/benchmark/wikidata/qald10/train.example-index.tar.gz
# extract example index
tar -xzf train.example-index.tar.gz
```

7. Run GRASP:

```bash
# With the config at configs/run.yaml, all important config options like model,
# function set, and knowledge graph can be set via env variables or directly
# in the config file. An example index for few-shot learning can be set via
# the KG_EXAMPLES env variable or also in the config file.
# See the config files for more details and other options.

# Note, that if you e.g. run OpenAI models, you also need to set the
# OPENAI_API_KEY env variable (see section about supported models below).

# --log-level DEBUG is recommended for more verbose output showing
# intermediate steps.

# Run GRASP on a question:
# By default, GRASP outputs the answer to stdout as JSON with some extra metadata.
# To avoid this we redirect it to /dev/null here, and set --log-level to DEBUG which
# shows all steps in a nicely formatted way.
grasp --log-level DEBUG run configs/run.yaml "Where was Angela Merkel born?" > /dev/null

# Run GRASP on a benchmark and save the output to a file, in this case QALD-10:
grasp --log-level DEBUG file configs/run.yaml \
  data/benchmark/wikidata/qald10/test.jsonl \
  --output-file data/benchmark/wikidata/qald10/outputs/test.jsonl

# Start a GRASP server, by default on port 8000:
grasp --log-level DEBUG serve configs/run.yaml

# For convenience, we also provide a config to run the server with all
# available knowledge graphs (make sure to download all indices first):
grasp --log-level DEBUG serve configs/serve.yaml
```

### Run evaluation app

Follow [these instructions](apps/evaluation/README.md) to run the evaluation app.

## Supported models

GRASP supports both commercial and open-source models.

### OpenAI

1. Set `OPENAI_API_KEY` env variable
2. Set model to `openai/<model_name>` in the config file or with
`MODEL` env variable, we tested:

- `openai/gpt-4.1`
- `openai/gpt-4.1-mini`
- `openai/o4-mini`
- `openai/gpt-5-mini`
- `openai/gpt-5`

### Google Gemini

1. Set `GEMINI_API_KEY`
2. Set model to `gemini/<model_name>` in the config file or with
`MODEL` env variable, we tested:

- `gemini/gemini-2.0-flash`
- `gemini/gemini-2.5-flash-preview-04-17`

### Local server with vLLM

1. Install vLLM with `pip install vllm`
2. Run vLLM server with a model of your choice, see below
3. Set model to `hosted_vllm/<model_name>` in the config file or with
`MODEL` env variable, we tested:

- `hosted_vllm/Qwen/Qwen2.5-72B-Instruct` (and other sizes)
- `hosted_vllm/Qwen/Qwen3-32B` (and other sizes)

4. Set model_endpoint in the config file or with `MODEL_ENDPOINT` env variable
to your vLLM server endpoint, by default this will be `http://localhost:8000/v1`

#### Run Qwen2.5

Change 72B to 7B, 14B, or 32B to run other sizes. Adapt the tensor parallel size
to your GPU setup, we used two H100 GPUs for Qwen2.7 72B.

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --tool-call-parser hermes \
--enable-auto-tool-choice --tensor-parallel-size 2
```

#### Run Qwen3

Change 32B to 4B, 8B, or 14B to run other sizes.

```bash
vllm serve Qwen/Qwen3-32B --enable-reasoning --reasoning-parser deepseek_r1 \
--tool-call-parser hermes --enable-auto-tool-choice
```

## Misc

To prepare some benchmark datasets with the [Makefile](Makefile),
e.g. using `make wikidata-benchmarks`, you first need to clone
[github.com/KGQA/KGQA-datasets](https://github.com/KGQA/KGQA-datasets) into `third_party`:

```bash
mkdir -p third_party
git clone https://github.com/KGQA/KGQA-datasets.git third_party/KGQA-datasets
```
