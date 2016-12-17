#!/bin/bash

if [ ! -d data_bi ]; then
	unzip data_bi
fi
if [ ! -d data_bi/result ]; then
	mkdir data_bi/result
fi

cd ..
make clean
make

data_folder=demo/data_bi/
result_folder=demo/data_bi/result/
edge_file=$data_folder/network.txt
node_file=$data_folder/node.txt
train_p2a_file=$data_folder/train_target.txt
train_p2o_file=$data_folder/train_feat.txt
test_p2a_file=$data_folder/test_target.txt
test_p2o_file=$data_folder/test_feat.txt

./main -network $edge_file -node2type $node_file -train $train_p2a_file -train_f $train_p2o_file -test $test_p2a_file -test_f $test_p2o_file -binary 0 -size 32 -negative 5 -samples 10 -map_topk 5 -net_loss 0 -supf_loss 0 -supf_negative_by_sampling 0. -supf_neg_sampling_pow 0. -supf_neg_base_deg 0 -threads 11 -omega 0.3 -lr_emb 0.1 -reg_emb 0. -supf_dropout 0. -pred $result_folder/pred.txt -output $result_folder/emb.txt

