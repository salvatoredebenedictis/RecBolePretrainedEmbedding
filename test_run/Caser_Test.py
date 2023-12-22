import sys
sys.path.append('../')
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import Caser
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import torch
from torch import nn

dict_params = {
    "gpu_id": '0',
    "D_PRODUCT_FIELD": "D_PRODUCT",
    "device": "cuda",
    "ITEM_ID_FIELD": "Key_product",
    "K_PRODUCT_TYPE_FIELD": "K_PRODUCT_TYPE",
    "Key_receipt_FIELD": "Key_receipt",
    "LINE_NUM_FIELD": "LINE_NUM",
    "LIST_SUFFIX": "_list",
    "MAX_ITEM_LIST_LENGTH": 20,
    "QUANTITY_FIELD": "QUANTITY",
    "Q_AMOUNT_FIELD": "Q_AMOUNT",
    "Q_DISCOUNT_AMOUNT_FIELD": "Q_DISCOUNT_AMOUNT",
    "TIME_FIELD": "TS_T_RECEIPT",
    "USER_ID_FIELD": "K_MEMBER",
    "data_path": "/kaggle/input/receipt-lines",
    "embedding_size": 64,
    "dataset": "RECEIPT_LINES",
    "dropout_prob": 0.3,
    "epochs": 100,
    "eval_args": {
        "group_by": "Key_receipt",
        "mode": "full",
        "order": "TO",
        "split": {
            "LS": "valid_and_test"
        }
    },
    "eval_batch_size": 1024,
    "hidden_size": 128,
    "load_col": {
        "inter": [
            "K_MEMBER",
            "TS_T_RECEIPT",
            "LINE_NUM",
            "Key_receipt",
            "QUANTITY",
            "Q_AMOUNT",
            "Q_DISCOUNT_AMOUNT",
            "Key_product"
        ],
        "item": [
            "D_PRODUCT",
            "K_PRODUCT_TYPE",
            "Key_product"
        ]
    },
    "loss_type": "CE",
    "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision","TailPercentage","AveragePopularity","ItemCoverage"],
    "num_layers": 1,
    "topk": 10,
    "train_batch_size": 4096,
    "train_neg_sample_args": None,
    "valid_metric": "MRR@10"
}


config = Config(model='Caser', dataset='RECEIPT_LINES', config_dict = dict_params)

# init random seed
init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

# write config info into log
logger.info(config)


dataset = create_dataset(config)
logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)


# model loading and initialization
model = Caser(config, train_data.dataset).to(config['device'])
logger.info(model)

# trainer loading and initialization
trainer = Trainer(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)