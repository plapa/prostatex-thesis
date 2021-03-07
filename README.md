ProstateX
==============================

Trying to create a novel way of predicting prostate cancer for the challenge [PROSTATEx Challenge 2017](https://wiki.cancerimagingarchive.net/display/Public/PROSTATEx+Challenge+2017) using Conditional Random Fields

## Accompanying papers

*  [Lapa, P.; Castelli, M.; Gonçalves, I.; Sala, E.; Rundo, L. A Hybrid End-to-End Approach Integrating Conditional Random Fields into CNNs for Prostate Cancer Detection on MRI. Appl. Sci. 2020, 10, 338. https://doi.org/10.3390/app10010338](https://www.mdpi.com/2076-3417/10/1/338#)
*   [Paulo Lapa, Ivo Gonçalves, Leonardo Rundo, and Mauro Castelli. 2019. Semantic learning machine improves the CNN-Based detection of prostate cancer in non-contrast-enhanced MRI. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '19). Association for Computing Machinery, New York, NY, USA, 1837–1845. DOI:https://doi.org/10.1145/3319619.3326864](https://dl.acm.org/doi/abs/10.1145/3319619.3326864)
*   [Paulo Lapa, Ivo Gonçalves, Leonardo Rundo, and Mauro Castelli. 2019. Enhancing classification performance of convolutional neural networks for prostate cancer detection on magnetic resonance images: a study with the semantic learning machine. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '19). Association for Computing Machinery, New York, NY, USA, 381–382. DOI:https://doi.org/10.1145/3319619.3322035](https://dl.acm.org/doi/abs/10.1145/3319619.3322035)
*   [Conditional random fields improve the CNN-based prostate cancer classification performance, Master Thesis, 2019](https://run.unl.pt/handle/10362/89470)


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
