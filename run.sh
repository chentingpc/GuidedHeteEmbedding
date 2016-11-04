#!/bin/bash
make clean
make

# this script is out-dated.
# see exp.sh

data_folder=~/pbase/x/R2016/10_MIMIC_III/data/our_neonates/
data_folder=~/pbase/x/R2016/10_MIMIC_III/data/our_neonates_symp/
result_folder=~/pbase/x/R2016/10_MIMIC_III/result/

edge_file=$data_folder/train/Edge.txt
node_file=$data_folder/train/Node.txt
task=task_diagnoses
task=task_drug
train_p2a_file=$data_folder/train/$task/patient2target.txt
train_p2o_file=$data_folder/train/$task/patient2other.txt
test_p2a_file=$data_folder/test/$task/patient2target.txt
test_p2o_file=$data_folder/test/$task/patient2other.txt


K=10
    ./main -network $edge_file -node2type $node_file -train $train_p2a_file -train_f $train_p2o_file -test $test_p2a_file -test_f $test_p2o_file -binary 0 -size 32 -negative 5 -samples 500 -map_topk $K -net_loss 0 -supf_loss 0 -supf_negative_by_sampling 1 -supf_neg_sampling_pow 0. -supf_neg_base_deg 1 -threads 11 -omega 0.2 -lr_net_emb 0.1 -lr_net_w 0. -lr_net_etype_bias 0.001 -lr_supf_emb 0.1 -lr_supf_ntype_w 0.001 -lr_supf_nbias 0.00 -reg_net_emb 0. -reg_supf_emb 0. -path_conf_file path_conf_file.txt

#./main -network $edge_file -node2type $node_file -train $train_p2a_file -train_f $train_p2o_file -test $test_p2a_file -test_f $test_p2o_file -binary 0 -size 32 -negative 5 -samples 500 -map_topk $K -net_loss 0 -supf_loss 0 -supf_negative_by_sampling 1 -supf_neg_sampling_pow 0. -supf_neg_base_deg 1 -threads 11 -omega 0.2 -lr_net_emb 0.1 -lr_net_w 0. -lr_net_etype_bias 0.001 -lr_supf_emb 0.1 -lr_supf_ntype_w 0.001 -lr_supf_nbias 0.00 -reg_net_emb 0. -reg_supf_emb 0. -path_conf_file path_conf_file.txt #-output $result_folder/emb.txt -pred $result_folder/pred.txt # -input $result_folder/emb.txt


<<comment
# random selection
while true ; do
    date=$(date)
    echo $date
    ./main -network $edge_file -node2type $node_file -train $train_p2a_file -train_f $train_p2o_file -test $test_p2a_file -test_f $test_p2o_file -binary 0 -size 32 -negative 5 -samples 100 -map_topk $K -net_loss 0 -supf_loss 0 -supf_negative_by_sampling 1 -supf_neg_sampling_pow 0. -supf_neg_base_deg 1 -threads 11 -omega 0.2 -lr_net_emb 0.1 -lr_net_w 0. -lr_net_etype_bias 0.001 -lr_supf_emb 0.1 -lr_supf_ntype_w 0.001 -lr_supf_nbias 0.00 -reg_net_emb 0. -reg_supf_emb 0. -path_conf_file path_conf_file.txt > results_path_selection/"$date"
done
comment

#rho=>lr_net_emb
#eta=>lr_net_w
#sigma=>lr_net_etype_bias
#lambda=>reg_net_emb

#epsilon=>lr_supf_emb
#mu=>lr_supf_ntype_w
#??=>lr_supf_nbias
#gamma=>reg_supf_emb

