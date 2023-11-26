device=0
data=data/simulated_data/
batch=20
n_head=2
n_layers=2
d_model=32
d_rnn=24
d_inner=64
d_k=64
d_v=64
dropout=0.1
lr=1e-4
smooth=0.1
epoch=10
log=log.txt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log
