import numpy as np
import itertools
import matplotlib.pyplot as plt
import random
import argparse
import os

np.random.seed(128)

class Logic_Model_Generator:

    def __init__(self, time_tolerance, decay_rate, time_horizon, sep):
        ### the following parameters are used to manually define the logic rules
        
        self.num_formula = 1 # num of prespecified logic rules
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.time_horizon = time_horizon
        self.time_tolerance = time_tolerance
        self.mental_predicate_set = [0]
        self.action_predicate_set = [1]
        self.body_predicate_set = [1]
        self.head_predicate_set = [0] # the index set of all head predicates
        self.num_predicate = len(self.mental_predicate_set + self.action_predicate_set)
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
        self.model_parameter[head_predicate_idx]['base'] = 0.01

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.4]
        # self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = [0.01, 0.01]
        
        # now, weight = "weight_para"[0] + "weight_para"[1] * cur_t

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

        # NOTE: rule content: x_1 --> x_0, x_1 before x_0
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 0]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]
        
        '''
        Action predicates: [1], x_1
        '''

        return logic_template

    def intensity(self, cur_time, head_predicate_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []

        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            # range all the formula for the chosen head_predicate
            cur_weight = self.model_parameter[head_predicate_idx][formula_idx]['weight_para'][0]
            # cur_weight = self.model_parameter[head_predicate_idx][formula_idx]['weight_para'][0] + \
            #              self.model_parameter[head_predicate_idx][formula_idx]['weight_para'][1] * cur_time
            
            
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
        
        # print('----- occur_time_dic -----')
        # print(occur_time_dic)
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
                    temporal_kernel *= (time_difference < -self.time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'EQUAL':
                    temporal_kernel *= (abs(time_difference) <= self.time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'AFTER':
                    temporal_kernel *= (time_difference > self.time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[1]]))
            # print('----- temporal_kernel -----')
            # print(temporal_kernel)
            feature = np.sum(temporal_kernel)
        
        # print('----- feature -----')
        # print(feature)
        return feature

    def get_formula_effect(self, template):
        '''
        get_formula_effect(): the body condition will boost the head to be 1 or 0 (positive or neg effect to head predicate) 
        (self.model_parameter[head_predicate_idx][formula_idx]['weight'] represents the effect's degree)
        '''
        if template['head_predicate_sign'][0] == 1:
            formula_effect = 1
        else:
            formula_effect = -1
        return formula_effect

    def sample_poisson(self):
        lam = random.choice(np.array([2, 3, 4]))
        num_events = 80
        # print('num_events', num_events)
        events = self.time_horizon * np.random.exponential(1 / lam, size=num_events)
        events.sort()
        new_events = []
        for item in events.tolist():
            if item <= self.time_horizon:
                new_events.append(item)
        return new_events

    def generate_data(self, num_sample, time_horizon):
        data = {}
        t_list_dict = {}
        intensity_list_dict = {}
        occur_t_list_dict = {}
        ratio_list_dict = {}
        occurred_m_ratio_list_dict = {}
        occurred_m_intensity_list_dict = {}
        # NOTE: data = {0:{}, 1:{}, ...., num_sample:{}}
        for sample_ID in np.arange(0, num_sample, 1):
            # print('---------- Start generating the {}-th sample ----------'.format(sample_ID))
            data[sample_ID] = {} # each data[sample_ID] stores one realization of the point process
            
            ##### initialize data
            ##### 对于当前这个简单版本, body predicate直接从poisson分布sample, head predicate根据intensity生成
            for predicate_idx in self.head_predicate_set:
                data[sample_ID][predicate_idx] = {}
                data[sample_ID][predicate_idx]['time'] = []
            
            for predicate_idx in self.body_predicate_set:
                data[sample_ID][predicate_idx] = {}
                data[sample_ID][predicate_idx]['time'] = self.sample_poisson()

            t = 0 # cur_time

            
            t_list = []
            intensity_list = []
            occur_t_list = []
            occur_intensity_list = []
            ratio_list = [] # we now consider ratio as ground truth hazard function in each grid
            occurred_m_ratio_list = []
            while t < time_horizon:
                grid = np.arange(t, time_horizon, self.sep)
                intensity_potential = [self.intensity(cur_time, self.head_predicate_set[0], data[sample_ID]) for cur_time in grid]
                intensity_max = np.max(np.array(intensity_potential))
                time_to_event = np.random.exponential(1 / intensity_max) # sample the interevent time

                # below check whether to accept above sampled time_to_event
                t = t + time_to_event
                ##### 保证生成的mental event时间点都小于time horizon
                if t < time_horizon:
                
                    # TODO: check whether keep min()
                    ratio = min(self.intensity(t, self.head_predicate_set[0], data[sample_ID]) / intensity_max, 1)
                    ratio_list.append(ratio)
                    # todo: ratio list as ground truth hazard rate?
                    ##########
                    t_list.append(t)
                    intensity_list.append(self.intensity(t, self.head_predicate_set[0], data[sample_ID]))
                    ##########
                
                    # print(sum([self.intensity(t, head_idx, data[sample_ID]) for head_idx in self.head_predicate_set]) / intensity_max)
                    flag = np.random.binomial(1, p=ratio)
                    if flag == 1:

                        # then decide which predicate is going to be triggered at the next event occur time, sample by their intensity
                        p = np.array( [self.intensity(t, head_idx, data[sample_ID]) for head_idx in self.head_predicate_set]) \
                                  / np.sum(np.array([self.intensity(t, head_idx, data[sample_ID]) for head_idx in self.head_predicate_set])
                                )
                        tmp = np.random.multinomial(1, pvals=p)
                        idx = self.head_predicate_set[np.argmax(tmp)]
                        data[sample_ID][idx]['time'].append(t)
                        occur_t_list.append(t)
                        occurred_m_ratio_list.append(ratio)
                        occur_intensity_list.append(self.intensity(t, self.head_predicate_set[0], data[sample_ID]))

                    else:
                        continue
                
                else:
                    break
            t_list_dict[sample_ID] = t_list
            intensity_list_dict[sample_ID] = intensity_list
            occur_t_list_dict[sample_ID] = occur_t_list
            ratio_list_dict[sample_ID] = ratio_list
            occurred_m_ratio_list_dict[sample_ID] = occurred_m_ratio_list
            occurred_m_intensity_list_dict[sample_ID] = occur_intensity_list
            ##### plot ratio
            # print(t_list)
            # print(ratio_list)
            # print(occur_t_list)

            
            # plt.plot(t_list, intensity_list, color ='blue', label='intensity')
            # plt.scatter(occur_t_list, occur_intensity_list, marker='o', color='red')
            # plt.xlabel('time')
            # plt.ylabel('intensity')
            # plt.title('intensity of mental event')
            # plt.legend(loc='best')
            # plt.savefig('intensity_mental')

        return data, t_list_dict, intensity_list_dict, occur_t_list_dict, ratio_list_dict, occurred_m_ratio_list_dict, occurred_m_intensity_list_dict
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-time_tolerance', help='time tolerance for computing the feature', type=int, default=0)
#     parser.add_argument('-decay_rate', help='decay rate for computing the feature', type=int, default=1)
#     parser.add_argument('-num_seqs', help='total number of generated sequences', type=int, default=1)
#     parser.add_argument('-time_horizon', help='the time horizon of all sequences', type=int, default=100)
#     args = parser.parse_args()
#
#     logic_model_generator = Logic_Model_Generator(args.time_tolerance, args.decay_rate, args.time_horizon)
#     data = logic_model_generator.generate_data(args.num_seqs, args.time_horizon)
#
#     ##### save data
#     if not os.path.exists("./Synthetic_Data"):
#         os.makedirs("./Synthetic_Data")
#     path = os.path.join("./Synthetic_Data", 'dataset_{}seqs_{}T.npy'.format(args.num_seqs, args.time_horizon))
#     np.save(path, data)
#
# # load synthetic dataset
# data = np.load("./Synthetic_Data/dataset_1seqs_200T.npy", allow_pickle=True).item()
# print(data)
#
# # data, time_horizon: 10
#
# # logic template
# # {1: {0: {'body_predicate_idx': [0],
# #          'body_predicate_sign': [1],
# #          'head_predicate_sign': [1],
# #          'temporal_relation_idx': [[0, 1]],
# #          'temporal_relation_type': ['BEFORE']}}}
#
# # ground truth parameters
# # template: {head_predicate: {'base': b, rule_1: {'weight_para': [w_0, w_1]}}}
# # {1: {'base': 0.2, 0: {'weight_para': [0.3, 0.04]}}}
