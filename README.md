# Energy Forecast Benchmark Toolkit Examples

Example models and tutorials for the energy forecast benchmarking
toolkit [enfobench](https://github.com/attila-balint-kul/energy-forecast-benchmark-toolkit).

## Table of Contents

- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Creating a Model](#creating-a-model)
- [License](#license)

## Folder Structure

The folder structure is as follows:

```
├── README.md                   <- The top-level README for getting started.
├── data
│   ├── load.csv                <- Example load profile data.
│   └── covariates.csv          <- Example weather data to use as covariates.
│
├── models                      <- Example models each in their own subfolder.
│   ├── naive-seasonal-1d       <- Naive seasonal model with 1 day seasonality.
│   │   ├── src                 <- All source code for the model.
│   │   │   └── main.py         <- Entrypoint for the forecast server.
│   ├── Dockerfile              <- Example dockerfile for the model. 
│   └── requirements.txt        <- Requirements for your model to run.
│
├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering).
│   ├── 01. Univariate.ipynb    <- Simple univariate forecast model benchmarking example.
│   └── 02. Multivariate.ipynb  <- Multivariate forecast model benchmarking example.
│
└── requirements.txt            <- Overall requirements to run all the example notebooks.
```

## Requirements

To contribute models to the benchmark you will need to have docker installed.
Please follow the installation procedure for your platform at
the [docker website](https://www.docker.com/products/docker-desktop/).

## Getting Started

To get started, you can clone this repository and install the requirements:

```bash
git clone https://github.com/attila-balint-kul/energy-forecast-benchmark-examples.git
cd energy-forecast-benchmark-examples
```

Then you can install the requirements: (Recommended inside a virtual environment)

```bash
pip install -r requirements.txt
```

Then you can run the example notebooks in the `notebooks` folder.

## Creating a Model

To create a model, you can use the `models/naive-seasonal-1d` folder as a template.
If you follow the folder structure, have a requirements.txt file and all your source
code is inside the `src/` folder, then there generally no need to change the `Dockerfile`.

Once your model is ready, you can build the docker image:

```bash
docker build -t tag-that-identifies-the-model ./path/to/the/folder/containing/the/Dockerfile
```

Once you built an image you can run a container

Then you can run the docker image:

```bash
docker run -p 3000:3000 tag-that-identifies-the-model
```

Then you can test your model by using the `ForecastClient` class from the `enfobench` package.
(Example can be found inside the `01. Univariate.ipynb` notebook)

Once the model is tested, you can push it to any public docker registry (e.g. dockerhub).
Then you can contact us with the repository and model tag and we will add it to the benchmark.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) file.
