import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
import pickle
np.random.seed(1024)


class Logic_Model_Generator:

    def __init__(self, time_tolerance, decay_rate, sep):

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
        self.total_predicate_set = [0, 1, 2]
        self.decay_rate = decay_rate # decay kernel
        self.sep = sep

        ### the following parameters are used to generate synthetic data
        ### for the learning part, the following is used to claim variables
        ### self.model_parameter = {0:{},1:{},...,6:{}}
        self.model_parameter = {}

        '''
        mental
        '''

        head_predicate_idx = 0
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = 0.3 # ini:0.1 0.3240

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.2] #ini:0.09, 0.2975
        # now, weight = "weight_para"[0] + "weight_para"[1] * cur_t
        '''
        action
        '''
        head_predicate_idx = 1
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = 0.2 # 0.4, 0.2577

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.3] # 0.1, 0.2105


        head_predicate_idx = 2
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = 0.1 # 0.2, 0.1361

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.4] # 0.9, 0.4374

        # NOTE: set the content of logic rules
        self.logic_template = self.logic_rule()

    def logic_rule(self):
        '''
        This function encodes the content of logic rules
        logic_template = {0:{},1:{},...,6:{}}
        '''

        '''
        Only head predicates may have negative values, because we only record each body predicate's boosted time 
        (states of body predicates are always be 1) 
        body predicates must happen before head predicate in the same logic rule
        '''

        logic_template = {}

        '''
        Mental predicate: [0]
        '''

        head_predicate_idx = 0
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 1 and before(1, 0) to 0
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 0]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        '''
        Action predicates: [1, 2]
        '''

        head_predicate_idx = 1
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 2 and before(2, 1) to 1
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
        # print("head is : ", head_predicate_idx)
        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            # range all the formula for the chosen head_predicate
            cur_weight = self.model_parameter[head_predicate_idx][formula_idx]['weight_para'][0]
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
            occur_time = np.array(history[body_predicate_idx]['time'])
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
            # print(feature)
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
        # NOTE: data = {0:{},1:{},....,num_sample:{}}
        for sample_ID in np.arange(0, num_sample, 1):
            print('----- generate the {}-th sequence -----'.format(sample_ID))
            data[sample_ID] = {}  # each data[sample_ID] stores one realization of the point process
            # initialize data
            for predicate_idx in np.arange(0, self.num_predicate, 1):
                data[sample_ID][predicate_idx] = {}
                data[sample_ID][predicate_idx]['time'] = []
            t = 0  # cur_time
            while t < time_horizon:
                grid = np.arange(t, time_horizon, self.sep)
                intensity_potential = []
                for time in grid:
                    intensity_potential.append(
                        [self.intensity(time, head_predicate_idx, data[sample_ID]) for head_predicate_idx in
                         self.head_predicate_set])
                intensity_potential = [sum(item) for item in intensity_potential]
                intensity_max = np.max(np.array(intensity_potential))
                time_to_event = np.random.exponential(1 / intensity_max)  # sample the interevent time
                # below check whether to accept above sampled time_to_event
                t = t + time_to_event
                
                
                ##### 保证生成的mental event时间点都小于time horizon
                if t < time_horizon:
                    ratio = min(sum([self.intensity(t, head_idx, data[sample_ID]) for head_idx in self.head_predicate_set]) / intensity_max, 1)
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
                else: # current t, namely t + time_to_event, greater than time_horizon
                    break

        return data

    def process_data(self, num_sample, data):
        processed_data = []
        for sample_ID in np.arange(0, num_sample, 1):
            processed_data.append([])
            time, type = [], []
            for predicate_index in self.total_predicate_set:
                time += data[sample_ID][predicate_index]['time']
                type += [predicate_index] * len(data[sample_ID][predicate_index]['time'])
                
            ##### sort according to occurred time (list time)
            sorted_lists = sorted(zip(time, type))
            time, type = zip(*sorted_lists)
            time, type = list(time), list(type)
            
            last_event = 0
            for event_idx in range(len(time)):
                time_since_last_event = time[event_idx] - last_event
                processed_data[sample_ID].append({'idx_event': event_idx + 1, 
                                                  'time_since_last_event': time_since_last_event, 
                                                  'type_event': type[event_idx] + 1, 
                                                  'time_since_start': time[event_idx]})
                last_event = time[event_idx]
        
        return processed_data

##### -- main -- #####
if __name__ == "__main__":
    sep = 0.1
    time_tolerance = 0.1
    decay_rate = 0.8
    data_type = 'train'
    time_horizon = 50
    num_sample = 10000
    logic_model_generator = Logic_Model_Generator(time_tolerance, decay_rate, sep)
    data = logic_model_generator.generate_data(num_sample, time_horizon)
    processed_data = logic_model_generator.process_data(num_sample, data)
    
    
    ##### save data
    if not os.path.exists('./data/simulated_data'):
        os.makedirs('./data/simulated_data')
    path = os.path.join('./data/simulated_data', '{}_dataset_{}seqs_{}T.pkl'.format(data_type, num_sample, time_horizon))
    with open(path, 'wb') as file:
        pickle.dump(processed_data, file)

## load synthetic dataset
# path = './data/simulated_data/test_dataset_3seqs_10T.pkl'
# with open(path, 'rb') as file:
#     loaded_data = pickle.load(file)

# print(loaded_data)
