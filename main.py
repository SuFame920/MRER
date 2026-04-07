#!/usr/bin/env python

import os
import argparse
import yaml
import random
import string
import torch
from datetime import datetime
from attrdict import AttrDict
from loguru import logger
import warnings

from src.tools import (
    apply_dataset_overrides,
    update_config,
    set_seed,
    load_params_bert,
    enable_console_logging,
)
from src.trainer import MyTrainer 
from src.loader import make_supervised_data_module
import transformers
from src.model import TextClassification

warnings.filterwarnings('ignore')

class Template:
    def __init__(self, args):
        self.console_log = None
        try:
            # 鍔犺浇閰嶇疆鏂囦欢
            config = AttrDict(yaml.load(
                open('src/config.yaml', 'r', encoding='utf-8'),
                Loader=yaml.FullLoader
            ))
            config = apply_dataset_overrides(config)

            # 鏇存柊閰嶇疆
            for k, v in vars(args).items():
                setattr(config, k, v)
            config = update_config(config)

            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.target_dir = os.path.join(config.target_dir, run_stamp)
            os.makedirs(config.target_dir, exist_ok=True)

            # 璁剧疆妯″瀷淇濆瓨鍚嶇О
            random_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
            config.save_name = f"{config.model_name}_{random_str}_{config.seed}_{{}}.pt"
            self.console_log = enable_console_logging(
                lambda: os.path.join(config.target_dir, "terminal.log")
            )
            # Reproducibility is fixed to strict mode.
            set_seed(config.seed)
            config.device = torch.device(f'cuda:{config.cuda_index}' if torch.cuda.is_available() else 'cpu')
            if config.device.type == 'cuda':
                print(f"[Device] Using GPU: {config.device} - {torch.cuda.get_device_name(config.cuda_index)}")
            else:
                print("[Device] Using CPU")

            self.config = config
        except Exception:
            self.close()
            raise

    def forward(self):
        # 鍒濆鍖杢okenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.bert_path, 
            padding_side="right",
            use_fast=False
        )

        # 鍑嗗鏁版嵁
        self.train_loader, self.valid_loader, self.test_loader, self.config = \
            make_supervised_data_module(self.config, tokenizer)
        if self.config.model_name == 'bert':
            self.model = TextClassification(self.config, tokenizer).to(self.config.device)

        # 鍔犺浇浼樺寲鍣ㄧ瓑鍙傛暟
        self.config = load_params_bert(self.config, self.model, self.train_loader) 

        # 璁粌妯″瀷
        trainer = MyTrainer(self.model, self.config, self.train_loader, self.valid_loader, self.test_loader)
        trainer.train()

    def close(self):
        if hasattr(self, "console_log") and self.console_log is not None:
            self.console_log.stop()
            self.console_log = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert', help='model type')
    parser.add_argument('-cd', '--cuda_index', type=int, default=0, help='cuda device index')

    template = None
    try:
        template = Template(parser.parse_args())
        template.forward()
    finally:
        if template is not None:
            template.close()


