#!/usr/bin/env bash

wget https://www.dropbox.com/s/1rh10bbn1mmod88/lr_8e-06_pre0.pt?dl=1 -O save_model/bert_jap.pt
wget https://www.dropbox.com/s/wnsv5s983qm129h/albert_best.pkl?dl=1 -O save_model/albert.pt
wget https://www.dropbox.com/s/hstdctir9rzb3e7/double_8e-6.pkl?dl=1 -O save_model/bert_mul1.pt
wget https://www.dropbox.com/s/jjpgv22nglnpy5i/QA_9e-06.pt?dl=1 -O save_model/bert_mul2.pt

