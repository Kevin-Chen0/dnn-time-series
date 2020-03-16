# Deep Neural Networks (DNN) for time-series data
Turnkey and modular deep learning predictive modeling package for time-series data. It allows for univariate and multivariate time-series as well as single and multi-step forecasts. DNN models include RNNs, LSTMs, GRUs, CNNs, hybrids, and more.



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

Step 4) To run it locally, open [local_run.ipynb](https://github.com/Kevin-Chen0/dnn-time-series/blob/master/example/local_run.ipynb) and proceed to run all. It will use [local_config.yaml](https://github.com/Kevin-Chen0/dnn-time-series/blob/master/example/local_config.yaml) as parameters to customize the procedures at runtime. It double-checks whether you installed the latest `dnntime` (v0.3.9.3) and will install for you if not. Make sure that you have set the dataset file path in local_config.yaml and the local_config.yaml path in local_run.ipynb.

**NOTE:** It is highly recommended to run this package using a GPU. Although CPU may work on small-scale datasets of < 10,000 samples, it may encounter performance issues on any dataset larger than that, including the example datasets found here. If you do not have a GPU, you can skip Step 4) and move to Step 5) Google Colab.

Step 5) To run it on Google Colab, go to the Colab [page](https://colab.research.google.com/notebooks/intro.ipynb) and upload both [colab_run.ipynb](https://github.com/Kevin-Chen0/dnn-time-series/blob/master/example/colab_run.ipynb) and [colab_config.yaml](https://github.com/Kevin-Chen0/dnn-time-series/blob/master/example/colab_config.yaml).

**NOTE:** You can copy this notebook to run on any cloud notebook as long as you can customize how to store and extract the files from that cloud instance. 
