# Neural Motion Controller

## Setup

1. If `docker ps -a` does not list a container named `nemoco-container`, run `./activate-container.sh`
2. If the container already exists, run `docker start nemoco-container -i`
3. Once inside container, run your scripts or start the jupyter server with `start-server.py`

## Dataset Generation

1. Starting point are two csv files, one for the input data and one for the output data
2. Combine and clean them with `prepare_data.py`, put the results into `src/datasets/prepared_datasets`
3. Generate trainbale datasets with `generate_datasets.py`

## Training

Just run the `train.py` script, the only input you need is the dataset you previously generated.