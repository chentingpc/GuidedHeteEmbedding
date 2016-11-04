#!/bin/bash

if [ $(hostname) = dmg1 ]; then
    data_folder=~/dbase/competitions/outbrain/network/
    result_folder=~/dbase/competitions/outbrain/results/
    edge_file=$data_folder/network_pv.txt
    node_file=$data_folder/node_pv.txt
    cat common.h | sed 's/hash_table_size = [e0-9]*/hash_table_size = 5e7/' | sed 's/neg_table_size = [e0-9]*/neg_table_size = 8e8/' > ccommon.h && mv ccommon.h common.h
else
    data_folder=~/dbase_ext/competitions/outbrain/network/
    result_folder=~/dbase_ext/competitions/outbrain/results/
    edge_file=$data_folder/network.txt
    node_file=$data_folder/node.txt
    cat common.h | sed 's/hash_table_size = [e0-9]*/hash_table_size = 5e7/' | sed 's/neg_table_size = [e0-9]*/neg_table_size = 1e8/' > ccommon.h && mv ccommon.h common.h
fi
make clean
make

train_p2a_file=$data_folder/ttrain_target.txt
train_p2o_file=$data_folder/ttrain_feature.txt
test_p2a_file=$data_folder/tvalid_target.txt
test_p2o_file=$data_folder/tvalid_feature.txt

<<comment
train_p2a_file=$data_folder/train_target.txt
train_p2o_file=$data_folder/train_feature.txt
test_p2a_file=$data_folder/test_target.txt
test_p2o_file=$data_folder/test_feature.txt
comment

mapk=12


#####################
# Best run
#####################
./main -network $edge_file -node2type $node_file -train $train_p2a_file -train_f $train_p2o_file -test $test_p2a_file -test_f $test_p2o_file -binary 0 -size 64 -negative 5 -samples 100 -map_topk $mapk -net_loss 0 -supf_loss 0 -supf_negative_by_sampling 0.5 -supf_neg_sampling_pow 0. -supf_neg_base_deg 0 -threads 11 -omega 0. -lr_net_emb 0.1 -lr_net_w 0. -lr_net_etype_bias 0.001 -lr_supf_emb 0.1 -lr_supf_ntype_w 0.001 -lr_supf_nbias 0.001 -reg_net_emb 0. -reg_supf_emb 0. -supf_dropout 0. -path_conf_file path_conf_file_outbrain_bkp.txt #-pred $result_folder/pred.txt


#####################
# Parameter tunings
#####################
<<comment
for omega in 0 0.3 0.7 1.0 ; do
for lr_emb in 0.5 0.1 0.05 0.01; do
for lr_side in 0.01 0.001; do
for reg_emb in 0 0.001; do
    echo $omega, $lr_emb, $lr_side, $reg_emb
    ./main -network $edge_file -node2type $node_file -train $train_p2a_file -train_f $train_p2o_file -test $test_p2a_file -test_f $test_p2o_file -binary 0 -size 32 -negative 5 -samples 300 -map_topk $mapk -net_loss 0 -supf_loss 0 -supf_negative_by_sampling 0 -supf_neg_sampling_pow 0. -supf_neg_base_deg 0 -threads 11 -omega $omega -lr_net_emb $lr_emb -lr_net_w 0. -lr_net_etype_bias $lr_side -lr_supf_emb $lr_emb -lr_supf_ntype_w $lr_side -lr_supf_nbias $lr_si -reg_net_emb $reg_emb -reg_supf_emb $reg_emb > $result_folder/log+omega$omega+lr_emb$lr_emb+lr_side$lr_side+reg_emb$reg_emb #-pred $result_folder/pred.txt #-path_conf_file path_conf_file.txt
done
done
done
done
comment
