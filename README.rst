Initial steps
--------------------

1. ``conda install mamba -n base -c conda-forge``
2. ``mama env create -f env.yaml``
3. ``conda activate nmt_service``

Run ``python test.py`` to ensure everything works.





Project Organization
-------------------------------------------------------------------------------

.. code::

   ├── README.rst          <- The top-level readme for developers.
   │
   │
   ├── data
   │   ├── interim         <- Intermediate data that has been transformed.
   │   ├── processed       <- The final, canonical data sets for modeling.
   │   └── raw             <- The original, immutable data dump.
   │
   │
   ├── models_weights      <- Trained and serialized models, model predictions,
   │                          or model summaries.
   │
   ├── notebooks           <- Only local jupyter notebooks.
   │
   ├── src                 <- Source code for use in this project.
   │   ├── __init__.py     <- Makes src a Python package.
   │   │
   │   ├── inference       <- Scripts to use trained models on a real task
   │   │     
   │   ├── models          <- Scripts for model description.
   │   │
   │   ├── preprocessing   <- Scripts containing data loading code
   │   │
   │   └── train           <- Scripts to train models and then use trained
   │                          models to make predictions.
   |
   ├── workflow            <- Storage for ML-ops-related code (MLFlow, AirFlow, etc).
