# Project Descripton

This project includes three different subprojects:

-   A frequent subgraph mining project based on [gSpan](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1184038&casa_token=hII_fVgAeycAAAAA:bc095z3FKW0KFzRYjy1yIcNp7jIKEy9jXd_9cv4FxosnXtFOkIHTkcy0cS8VJqrUK52z7aHIzw&tag=1)
-   A discriminative subgraph mining project based on [GAIA](https://dl.acm.org/doi/pdf/10.1145/1807167.1807262?casa_token=G-mvu1_NCJgAAAAA:4ToovRUGv-i_YrmH8SxNQP15J3hTY-N0rq49oNgp1h45khNM6c3cgwcQjKGe66q-05jZeFXg8Kfr)
-   A `pytorch` environment that encodes features (preprocesses graph data) produced by `gSpan` and `GAIA` and trains on processed data

# How-to

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

-   Since we use TUD Benchmark, we need to convert TUD data to [Networkx](https://networkx.org/documentation/stable/index.html) data. To do so run the following command within the `pytorch` project
    ```
    # python utilities/convert.py -d {dataset_name}
    # for example if we want to convert MUTAG to Networkx, run the following
    python utilities/convert.py -d MUTAG
    ```
