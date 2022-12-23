Initial steps
--------------------

```
conda env create --file env.yaml
conda activate nmt_service
pip install python-telegram-bot
conda install -c conda-forge youtokentome
```


Run ``python test.py`` to ensure everything works.

Usage
--------------------
``python cli.py``


Script configurations
--------------------

**Preprocess**

``python preprocess.py --train-src data/raw/train.ru_en.ru --train-trg data/raw/train.ru_en.en --val-src data/raw/valid.ru_en.ru --val-trg data/raw/valid.ru_en.en --test-src data/raw/test.ru_en.ru --test-trg data/raw/test.ru_en.en --src-lang ru --trg-lang en --src-size 10000 --trg-size 10000 --save-data-dir ./data/processed/ --max-len 64 --src-model-path ./weights_models/ru.model --trg-model-path ./weights_models/en.model``

**Train**

``python train.py --train-path ./data/processed/train.ru_en.pkl --val-path ./data/processed/val.ru_en.pkl --embedding-size 512 --n-heads 8 --n-layers 3 --dropout 0.1 --lr 0.0002 --max-epochs 100 --batch-size 128 --src-vocab-size 10000 --trg-vocab-size 10000 --src-lang ru --trg-lang en --max-seq-len 64 --display-freq 100 --model-path ./weights_models/transformer.pt``

**Generate**

``python generate.py --model-path ./weights_models/transformer.pt --src-sentence "YOUR_EN_SENTENCE" --max-seq-len 64 --src-tokenizer-path ./weights_models/ru_tokenizer.model --trg-tokenizer-path ./weights_models/en_tokenizer.model --use-cuda True``

**Run telegram bot**

Before run a telegram bot the telegram bot token is need to be created and passed as --telegram-token argument.

``python telegram_bot.py --telegram-token "YOUR_TELEGRAM_TOKEN" --model-path ./weights_models/transformer.pt --max-seq-len 64 --src-tokenizer-path ./weights_models/ru_tokenizer.model --trg-tokenizer-path ./weights_models/en_tokenizer.model --use-cuda True``

Data location
-------
https://drive.google.com/drive/folders/1XfUL9D6jeMv1ylD4pw7OOparCKQCvXkQ?usp=sharing


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
