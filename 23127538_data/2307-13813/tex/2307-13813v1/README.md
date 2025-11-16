# TeX-flow
Template repository for paper TeX based paper writing.

# Getting started

## Upgrade your TeXlive installation

You only need to do this step if your distribution is older than 2021.

### Remove your old distribution
If your installation is the default `/usr/local/texlive/...`
```shell
$ bash
$ rm -R /usr/local/texlive/<year>/
```
otherwise use the appropriate path.
For details see [here](https://www.tug.org/mactex/uninstalling.html).

### Get the new distribution
Download the new installer [here](http://www.tug.org/mactex/mactex-download.html) and follow instructions.

## Install Rust
```shell
$ curl https://sh.rustup.rs -sSf | sh
```

## Setup python environment and activate
```shell
$ conda env create -f environment.yaml
$ conda activate tex-flow-py37
```

## Install pre-commit hook
```shell
$ pre-commit install
```
