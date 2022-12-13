Initial steps
--------------------

``conda env create --file env.yaml``


Run ``python test.py`` to ensure everything works.

Preprocess
python preprocess.py --train-src data/raw/train.ru_en.ru --train-trg data/raw/train.ru_en.en --val-src data/raw/valid.ru_en.ru --val-trg data/raw/valid.ru_en.en --test-src data/raw/test.ru_en.ru --test-trg data/raw/test.ru_en.en --src-lang ru --trg-lang en --src-size 4000 --trg-size 4000 --save-data-dir ./data/processed/ --max-len 128 --src-model-path ./weights_models/ru.model --trg-model-path ./weights_models/en.model

Train
python train.py --train-path ./data/processed/train.ru_en.pkl --val-path ./data/processed/val.ru_en.pkl --embedding-size 512 --n-heads 8 --n-layers 6 --dropout 0.1 --lr 0.0002 --max-epochs 10 --batch-size 64 --src-vocab-size 4000 --trg-vocab-size 4000 --src-lang ru --trg-lang en --max-seq-len 128 --display-freq 100 --model-path ./weights_models/transformer.pt




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
