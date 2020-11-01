# precarious-papers
Spatial Econometrics is a sub-field of economics and spatial analysis, aimed at exploring the phenomenon of spatial interactive in economic data.  While much of the work in this sub-field has looked to investigate the factors which influence house prices and unemployment  across  specific  geographies,  work  by  Manski  (1993)  has  extended  this  theory  to  the  more  general  study  of  social graphs.  While the application of Spatial Econometrics has seen interest inside the literature of Financial Economics (Kou et al.,2018; Fernandez, 2011), little applied work exists on the application of Social Network Econometrics to the Capital Asset Pricing Model and Arbitrage Pricing Theory.  In this work, we aim to explore the transmission of risk factors using data released by theInternational Consortium of Investigative Journalists (ICIJ) on the Paradise Papers, to investigate the impact of firm interaction on price.

![island-fire](logo.png)

## Overview

This is your new Kedro project, which was generated using `Kedro 0.15.6` by running:

```
kedro new
```

Take a look at the [documentation](https://kedro.readthedocs.io) to get started.


## Installing dependencies

Dependencies should be declared in `src/requirements.txt` for pip installation and `src/environment.yml` for conda installation.

To install them, run:

```
kedro install
```

For those looking to run the pipeline, a docker container as been provided for use in reproducing our analysis. This can be run to recreate our environment using:
```
docker-compose run python
```
This will assume all raw data sources have been downloaded

## Running Kedro

You can run your Kedro project with:

```
kedro run
```
This will assume a IEXCloud api key is provided in `config/local/secrets.yml` as:
```
iex: YOUR_KEY
```
For researchers using our downloaded data, they may use:
```
kedro run --tag=local --parallel
```
This will assume all data is already available in the data directory.


### Working with Kedro from notebooks
Our analysis does provide `py:percent` format notebook which provides discussion over our exploratory work. This can be run either directly in a notebook environment that support `py:percent`, like VSCode, or may be converted to `ipynb` files using:
```
jupytext --to ipynb *.py
```

In order to use notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

For using Jupyter Lab, you need to install it:

```
pip install jupyterlab
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

You can also start Jupyter Lab:

```
kedro jupyter lab
```

And if you want to run an IPython session:

```
kedro ipython
```

Running Jupyter or IPython this way provides the following variables in
scope: `proj_dir`, `proj_name`, `conf`, `io`, `parameters` and `startup_error`.


## Building API documentation
Project documentation has been provided to guide users through our code. This can be found at the `docs/build/html/index.html`.

To build API docs for your code using Sphinx, run:

```
kedro build-docs
```

See your documentation by opening `docs/build/html/index.html`.
