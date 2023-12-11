mental_type_list=0
action_type_list=12
device=0
time_tolerance=0.1
decay_rate=0.8
time_horizon=50
num_sample=50
sep_for_grids=0.1
sep_for_data_syn=0.1
d_emb=30
d_h=3
d_k=20
d_v=20
d_hid=40
dropout=0.1
batch_size=2
lr=1e-3
num_iter=1


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python main.py -mental_type_list $mental_type_list -action_type_list $action_type_list -time_tolerance $time_tolerance -decay_rate $decay_rate -time_horizon $time_horizon -num_sample $num_sample -sep_for_grids $sep_for_grids -sep_for_data_syn $sep_for_data_syn -d_emb $d_emb -d_h $d_h -d_k $d_k -d_v $d_v -d_hid $d_hid -dropout $dropout -batch_size $batch_size -lr $lr -num_iter $num_iter
