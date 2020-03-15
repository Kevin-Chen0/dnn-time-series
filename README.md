# Deep Neural Networks (DNN) for time-series data
Turnkey and modular deep learning predictive modeling package for time-series data. It allows for univarate and multivariate time-series as well as single and multi-step forecasts. DNN models includes RNNs, LSTMs, GRUs, CNNs, hybrids, and more.



### Quick start

Step 1) Create and activate new env using `pipenv` or `conda` with Python 3.6 or higher. Here, the env is named `dts`.

```
conda create -n dts python=3.6
conda activate dts
```

Step 2) Pip install `dnntime` package. It will automatically install or update the dependent packages.

```pip install dnntime```

Step 3) In your working directory, download the example directory from this repo and `cd` into it.

```
svn export https://github.com/Kevin-Chen0/dnn-time-series.git/trunk/example
cd example
```

Step 4) Open [local_run.ipynb](https://github.com/Kevin-Chen0/dnn-time-series/blob/master/example/local_run.ipynb) and proceed to run all. It will use [local_config.yaml](https://github.com/Kevin-Chen0/dnn-time-series/blob/master/example/local_config.yaml) as parameters to customize the procedures at runtime. If the latest `dnntime` (v0.3.9.3) is already installed, you do not need to reinstall it again.

**NOTE:** It is highly recommended to run this package using a GPU, such as on Google Colab. Although CPU may work on small-scale datasets of < 10,000 samples, it may encounter encounter performance issue on any dataset larger than that, including the example datasets found here.
