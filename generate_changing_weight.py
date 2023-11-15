import numpy as np
import itertools
import matplotlib.pyplot as plt
import os

np.random.seed(128)

class Logic_Model_Generator:

    def __init__(self, time_tolerance, decay_rate):

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 3
        self.num_formula = 3  # num of prespecified logic rules
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = time_tolerance
        self.mental_predicate_set = [0]
        self.action_predicate_set = [1, 2]
        self.head_predicate_set = [0, 1, 2]  # the index set of all head predicates
        self.decay_rate = decay_rate # decay kernel

        ### the following parameters are used to generate synthetic data
        ### for the learning part, the following is used to claim variables
        ### self.model_parameter = {0:{},1:{},...,6:{}}
        self.model_parameter = {}

        '''
        mental
        '''

        head_predicate_idx = 0
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = 0.3

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.2, 0.05]
        # self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.6]
        # now, weight = "weight_para"[0] + "weight_para"[1] * cur_t
        '''
        action
        '''
        head_predicate_idx = 1
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = 0.2

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.3, 0.04]
        # self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.5]

        head_predicate_idx = 2
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = 0.1

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.4, 0.05]
        # self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.4]
 
        # NOTE: set the content of logic rules
        self.logic_template = self.logic_rule()

    def logic_rule(self):
        '''
        This function encodes the content of logic rules
        logic_template = {0:{}, 1:{}, ..., 6:{}}
        '''

        '''
        Only head predicates may have negative values, because we only record each body predicate's boosted time 
        (states of body predicates are always be 1) 
        body predicates must happen before head predicate in the same logic rule
        '''

        logic_template = {}

        '''
        Mental predicate: [0], x_0
        '''

        head_predicate_idx = 0
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 1 and before(1, 0) to 0
        # NOTE: x_1 --> x_0, x_1 before x_0
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 0]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        '''
        Action predicates: [1, 2], x_1 and x_2
        '''

        head_predicate_idx = 1
        logic_template[head_predicate_idx] = {}

        # TODO "2 to 1" & "1 and 0 and before(1,0) to \neg 2" contradict ?
        # NOTE: rule content: 2 and before(2, 1) to 1
        # NOTE: x_2 --> x_1, x_2 before x_1
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        head_predicate_idx = 2
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 0 and before(0, 2) to 2
        # NOTE: x_0 --> x_2, x_0 before x_2
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 2]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        return logic_template

    def intensity(self, cur_time, head_predicate_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []

        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            # range all the formula for the chosen head_predicate
            cur_weight = self.model_parameter[head_predicate_idx][formula_idx]['weight_para'][0] + \
                         self.model_parameter[head_predicate_idx][formula_idx]['weight_para'][1] * cur_time
            # cur_weight = self.model_parameter[head_predicate_idx][formula_idx]['weight_para'][0]
            
            weight_formula.append(cur_weight)
            # print("---->> cur formula idx: ", formula_idx)
            # print("---->> cur feature is: ", self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
            #                                         history=history,
            #                                         template=self.logic_template[head_predicate_idx][formula_idx]))
            feature_formula.append(self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                    history=history,
                                                    template=self.logic_template[head_predicate_idx][formula_idx]))
            effect_formula.append(self.get_formula_effect(template=self.logic_template[head_predicate_idx][formula_idx]))

        intensity = np.array(weight_formula) * np.array(feature_formula) * np.array(effect_formula)
        intensity = self.model_parameter[head_predicate_idx]['base'] + np.sum(intensity)
        # print("INTENSITY before transform: ")
        # print(intensity)
        if intensity >= 0:
            intensity = np.max([intensity, self.model_parameter[head_predicate_idx]['base']])
        else:
            # TODO: in this case, the intensity with respect to neg effect will always be positive,
            #  and it maybe even bigger than some intensity correspond to positive effect
            intensity = np.exp(intensity)
        return intensity

    def get_feature(self, cur_time, head_predicate_idx, history, template):
        occur_time_dic = {}
        feature = 0
        for idx, body_predicate_idx in enumerate(template['body_predicate_idx']):
            occur_time = np.array(history[body_predicate_idx]['time'][1:])
            mask = (occur_time <= cur_time)  # find corresponding history
            # mask: filter all the history time points that satisfies the conditions which will boost the head predicate
            occur_time_dic[body_predicate_idx] = occur_time[mask]
        occur_time_dic[head_predicate_idx] = [cur_time]
        ### get weight
        # compute features whenever any item of the transition_item_dic is nonempty
        history_transition_len = [len(i) for i in occur_time_dic.values()]
        if min(history_transition_len) > 0:
            # need to compute feature using logic rules
            time_combination = np.array(list(itertools.product(*occur_time_dic.values())))
            # get all possible time combinations
            time_combination_dic = {}
            for i, idx in enumerate(list(occur_time_dic.keys())):
                time_combination_dic[idx] = time_combination[:, i]
            temporal_kernel = np.ones(len(time_combination))
            for idx, temporal_relation_idx in enumerate(template['temporal_relation_idx']):
                time_difference = time_combination_dic[temporal_relation_idx[0]] - time_combination_dic[temporal_relation_idx[1]]
                if template['temporal_relation_type'][idx] == 'BEFORE':
                    temporal_kernel *= (time_difference < -self.Time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'EQUAL':
                    temporal_kernel *= (abs(time_difference) <= self.Time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'AFTER':
                    temporal_kernel *= (time_difference > self.Time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[1]]))
            feature = np.sum(temporal_kernel)
        return feature

    '''
    get_formula_effect(): the body condition will boost the head to be 1 or 0 (positive or neg effect to head predicate) 
    (self.model_parameter[head_predicate_idx][formula_idx]['weight'] represents the effect's degree)
    '''

    def get_formula_effect(self, template):
        if template['head_predicate_sign'][0] == 1:
            formula_effect = 1
        else:
            formula_effect = -1
        return formula_effect

    def generate_data(self, num_sample, time_horizon):
        data = {}

        # NOTE: data = {0:{}, 1:{}, ...., num_sample:{}}
        for sample_ID in np.arange(0, num_sample, 1):
            print('---------- Start generating the {}-th sample ----------'.format(sample_ID))
            data[sample_ID] = {} # each data[sample_ID] stores one realization of the point process
            # initialize data
            for predicate_idx in np.arange(0, self.num_predicate, 1):
                data[sample_ID][predicate_idx] = {}
                # TODO: 这里第一个位置的0是否需要加入？？？
                # data[sample_ID][predicate_idx]['time'] = [0]
                data[sample_ID][predicate_idx]['time'] = []
                
            
            t = 0  # cur_time
            sep = 0.03
            while t < time_horizon:
                grid = np.arange(t, time_horizon, sep)
                intensity_potential = []
                for time in grid:
                    intensity_potential.append(
                        [self.intensity(time, head_predicate_idx, data[sample_ID]) for head_predicate_idx in
                         self.head_predicate_set])
                intensity_potential = [sum(item) for item in intensity_potential]
                intensity_max = np.max(np.array(intensity_potential))
                time_to_event = np.random.exponential(1 / intensity_max) # sample the interevent time
                # below check whether to accept above sampled time_to_event
                t = t + time_to_event
                # TODO: check whether keep min()
                ratio = min(sum([self.intensity(t, head_idx, data[sample_ID]) for head_idx in self.head_predicate_set]) / intensity_max, 1)
                # print(sum([self.intensity(t, head_idx, data[sample_ID]) for head_idx in self.head_predicate_set]) / intensity_max)
                flag = np.random.binomial(1, p=ratio)
                if flag == 1:
                    # then decide which predicate is going to be triggered at the next event occur time
                    tmp = np.random.multinomial(1, pvals=np.array(
                        [self.intensity(t, head_idx, data[sample_ID]) for head_idx in self.head_predicate_set])
                                                         / np.sum(np.array(
                        [self.intensity(t, head_idx, data[sample_ID]) for head_idx in self.head_predicate_set])))
                    idx = np.argmax(tmp)
                    data[sample_ID][idx]['time'].append(t)
                else:
                    continue

        return data

    # def compute_intensity(self, data, time_horizon, step=0.03):
    #     intensity = {}
    #     time_grid = np.arange(0, time_horizon, step)
    #     for sample_ID in data:
    #         intensity[sample_ID] = {}
    #         for predicate_idx in self.head_predicate_set:
    #             intensity[sample_ID][predicate_idx] = []
    #     for sample_ID in data:
    #         for t in time_grid:
    #             for predicate_idx in self.head_predicate_set:
    #                 intensity[sample_ID][predicate_idx].append(
    #                     self.intensity(t, predicate_idx, history=data[sample_ID]))
    #     return intensity
    
if __name__ == "__main__":
    time_tolerance = 0.1
    decay_rate = 1
    logic_model_generator = Logic_Model_Generator(time_tolerance, decay_rate)
    data = logic_model_generator.generate_data(num_sample=3, time_horizon=10)
    
    ##### save data
    if not os.path.exists("./Synthetic_Data"):
        os.makedirs("./Synthetic_Data")
    path = os.path.join("./Synthetic_Data", 'dataset_3seqs_changing_weights.npy')
    np.save(path, data)

# load synthetic dataset
data = np.load("./Synthetic_Data/dataset_3seqs_changing_weights.npy", allow_pickle=True).item()
print(data)

# data, time_horizon: 10
# {0: {0: {'time': [3.8128389397850686, 5.459980388252364, 5.685251811136125, 
#                   7.018236445496469, 10.80601675167377]}, 
#      1: {'time': [3.3578014443304394, 5.346011181436626]}, 
#      2: {'time': []}}, 
#  1: {0: {'time': [0.22383193443832397, 4.693866428132608]}, 
#      1: {'time': [2.4136290095366566, 9.512258982538434, 9.94617597266919]}, 
#      2: {'time': [2.652982367418805, 3.5141145906260474, 7.790785570279592, 
#                   10.965734824921865]}}, 
#  2: {0: {'time': [0.375085940475808, 3.2880661117279955, 4.023014403073016, 
#                   5.07868636436225, 5.618894918265917, 5.945294987023819, 
#                   7.4253341741921535]}, 
#      1: {'time': [2.9398193946777518, 3.467780166750448, 8.371159219431188, 
#                   8.65890734280003, 10.161824737490626]}, 
#      2: {'time': [4.465806760140467, 5.12758732338402, 9.599508336946428]}}}

# ground truth parameters
# {0: {'base': tensor([0.3]), 
#      0: {'weight_para': tensor([0.2, 0.05])}}, 
#  1: {'base': tensor([0.2]), 
#      0: {'weight_para': tensor([0.3, 0.04])}}, 
#  2: {'base': tensor([0.1]), 
#      0: {'weight_para': tensor([0.4, 0.05]])}}}

# learned model parameters
# {0: {'base': tensor([0.3245], dtype=torch.float64, requires_grad=True), 
#      0: {'weight_para': tensor([0.0677, 0.0598], dtype=torch.float64, requires_grad=True)}}, 
#  1: {'base': tensor([0.2108], dtype=torch.float64, requires_grad=True), 
#      0: {'weight_para': tensor([0.1212, 0.0718], dtype=torch.float64, requires_grad=True)}}, 
#  2: {'base': tensor([0.1158], dtype=torch.float64, requires_grad=True), 
#      0: {'weight_para': tensor([0.1496, 0.0906], dtype=torch.float64, requires_grad=True)}}}
