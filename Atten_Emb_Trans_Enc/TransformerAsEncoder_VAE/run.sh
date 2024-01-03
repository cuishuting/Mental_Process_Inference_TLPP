mental_type_list=1
action_type_list=23
head_predicates_list=123
device=0
time_tolerance=0.1
decay_rate=0.8
time_horizon=60
num_sample=100
sep_for_grids=1.0
sep_for_data_syn=0.1
d_emb=300
d_h=3
d_hid=400
dropout=0.1
num_sublayer=1
batch_size=10
lr=1e-4
lr_scheduler_step=100
num_iter=5
tau_temperature=1.0
integral_sep=0.03


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python main.py -mental_type_list $mental_type_list -action_type_list $action_type_list -head_predicates_list $head_predicates_list -time_tolerance $time_tolerance -decay_rate $decay_rate -time_horizon $time_horizon -num_sample $num_sample -sep_for_grids $sep_for_grids -sep_for_data_syn $sep_for_data_syn -d_emb $d_emb -d_h $d_h -d_hid $d_hid -dropout $dropout -num_sublayer $num_sublayer -batch_size $batch_size -lr $lr -lr_scheduler_step $lr_scheduler_step -num_iter $num_iter -tau_temperature $tau_temperature -integral_sep $integral_sep
