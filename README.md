# Project Descripton

This project includes three different subprojects:

-   A frequent subgraph mining project based on [gSpan](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1184038&casa_token=hII_fVgAeycAAAAA:bc095z3FKW0KFzRYjy1yIcNp7jIKEy9jXd_9cv4FxosnXtFOkIHTkcy0cS8VJqrUK52z7aHIzw&tag=1)
-   A discriminative subgraph mining project based on [GAIA](https://dl.acm.org/doi/pdf/10.1145/1807167.1807262?casa_token=G-mvu1_NCJgAAAAA:4ToovRUGv-i_YrmH8SxNQP15J3hTY-N0rq49oNgp1h45khNM6c3cgwcQjKGe66q-05jZeFXg8Kfr)
-   A `pytorch` environment that encodes features (preprocesses graph data) produced by `gSpan` and `GAIA` and trains on processed data

# Setup

-   gSpan setup
    -   Create a virtual environment `env` within `gSpan` project by running `python3 -m venv env`.
    -   Install packages by running `pip install -r requirements.txt`
    -   Replace the files `gspan.py` and `graph.py` at `env/lib/python3/site-packages/gspan_mining/` with the files provided at `gspan/nx_support/`
-   GAIA setup
    -   Download link and instruction of GAIA can be found [here](https://sourceforge.net/projects/discriminatives/)
-   PyTorch Environment
    -   Create a virtual environment `env` within `pytorch` project by running `python3 -m venv env`.
    -   Install packages by running `pip install -r requirements.txt`

# Experiments

## Generating Relational Features

### Preparing data for gSpan

Since we use TUD Benchmark, we need to convert TUD data to [Networkx](https://networkx.org/documentation/stable/index.html) data. To do so run the following command within the `pytorch` project.

```bash
# python utilities/convert.py -d {dataset_name}
# for example if we want to convert MUTAG to Networkx, run the following
python utilities/convert.py -d MUTAG
```

This command will convert TUD to Networkx and it will save a `{dataset}.dat` file under `pytorch/data_networkx/{dataset}.dat`.

Next, copy the generated `{dataset}.dat` file to `gspan/data_nx/{dataset}/{dataset}.dat`. Well-known dataset `MUTAG` already exists as an example.

Next we have to convert `networkx` data to a specific input that can be used with `gSpan`. To do so run the following command within the `gSpan` project.

```bash
# python utilities/nx_to_graph.py -d {dataset_name}
# for example if we want to convert MUTAG to Networkx, run the following
python utilities/nx_to_graph.py -d MUTAG
```

This will produce a `{dataset}.graph` file under `gspan/data_graph/{dataset}.graph`.

### Mining Subgraphs with gSpan

Now that we have `.graph` data we can generate relational features by running the following command within the `gspan` project.

```bash
# python app.py -d {dataset_name} -s {support_in_percentage} -n {length_of_dataset/nr_of_graphs}
# for example if we want to generate subgraphs of MUTAG with 90%, run the following
python app.py -d MUTAG -s 90 -n 188
```

This command should generate 13 subgraphs and save them under `/data_nx_features/{dataset}.dat`.

###
