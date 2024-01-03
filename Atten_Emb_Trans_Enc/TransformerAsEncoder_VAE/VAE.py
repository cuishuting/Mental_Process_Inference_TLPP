import torch.nn as nn
from torch.nn.functional import softmax, gumbel_softmax
from Utils import logic_rule
import torch
import numpy as np
import itertools


class EncoderDecoder(nn.Module):

    def __init__(self, transformer_encoder, transformer_decoder, action_embed, mental_embed, generator, mental_sampler, rule_based_decoder):
        super(EncoderDecoder, self).__init__()
        self.transformer_encoder = transformer_encoder
        self.transformer_decoder = transformer_decoder
        self.action_embed = action_embed
        self.mental_embed = mental_embed
        self.generator = generator
        self.mental_sampler = mental_sampler
        self.rule_based_decoder = rule_based_decoder

    def encode(self, src, src_mask):
        return self.transformer_encoder(self.action_embed(src), src_mask)

    def decode(self, memory, tgt, tgt_mask):
        return self.transformer_decoder(self.mental_embed(tgt), memory, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences"""
        return self.decode(self.encode(src, src_mask), tgt, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + logsoftmax generation step for each grid's hazard func."""

    def __init__(self, d_model, num_mental_types):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, num_mental_types)

    def forward(self, x):
        # todo: an important advantage about using log_softmax in addition to numerical stability: this activation
        #  function heavily penalizes wrong class prediction as compared to its Softmax counterpart
        # return log_softmax(self.proj(x), dim=-1)
        logits = softmax(self.proj(x), dim=-1)  # consider output[0] as hz of mental 0
        return logits


class MentalSampler(nn.Module):
    def __init__(self, temperature):
        super(MentalSampler, self).__init__()
        self.tau = temperature

    def forward(self, logits): # logits shape: [batch_size, num_grids, num_mental_types]
        mental_samples = gumbel_softmax(logits, tau=self.tau, hard=True)
        return mental_samples


class LogicRuleBasedDecoder(nn.Module):
    def __init__(self, time_tolerance, decay_rate, initial_rule_params, action_type_list, mental_type_list, head_predicates_list, num_rules_boost_each_head, integral_sep):
        # todo: action_type_list doesn't include none type
        """
        param initial_rule_params: is a list like: [0.1, [0.09], 0.4, [0.1], 0.2, [0.9]], where the scalar is
        the base intensity for each head_predicate, and sub_list are the weight parameters for all the rules whose head is
        current head_predicate.
        param num_rules_boost_each_head: is a dict like: {1: 1, 2: 1, 3: 1}, keys are the head_idx, values are crsp num_of_rules boosting head_idx
        """
        super(LogicRuleBasedDecoder, self).__init__()
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.time_tolerance = time_tolerance
        self.decay_rate = decay_rate
        self.action_type_list = action_type_list  # [2, 3]
        self.mental_type_list = mental_type_list  # [1]
        self.num_action_types = len(self.action_type_list)
        self.num_mental_types = len(self.mental_type_list)
        self.head_predicates_list = head_predicates_list  # [1, 2, 3]
        self.num_rules_boost_each_head =num_rules_boost_each_head
        self.integral_sep = integral_sep
        self.model_parameters = {}
        """mental"""
        for idx, head_idx in enumerate(self.mental_type_list):
            self.model_parameters[head_idx] = {}
            self.model_parameters[head_idx]['base'] = torch.tensor([initial_rule_params[idx*2]], dtype=torch.float64, requires_grad=True)
            for rule_idx in range(self.num_rules_boost_each_head[head_idx]):
                self.model_parameters[head_idx][rule_idx] = {}
                self.model_parameters[head_idx][rule_idx]["weight"] = torch.tensor(initial_rule_params[idx*2+1][rule_idx], dtype=torch.float64, requires_grad=True)

        """action"""
        for idx, head_idx in enumerate(self.action_type_list):
            self.model_parameters[head_idx] = {}
            self.model_parameters[head_idx]['base'] = torch.tensor([initial_rule_params[(self.num_mental_types+idx)*2]], dtype=torch.float64, requires_grad=True)
            for rule_idx in range(self.num_rules_boost_each_head[head_idx]):
                self.model_parameters[head_idx][rule_idx] = {}
                self.model_parameters[head_idx][rule_idx]["weight"] = torch.tensor(initial_rule_params[(self.num_mental_types+idx)*2+1][rule_idx], dtype=torch.float64, requires_grad=True)
        self.logic_template = logic_rule()

    def forward(self, history_a, history_m):
        """
        history_a: (history_a_time, history_a_type), tuple
        history_a_time&history_a_type: tensor with shape [batch_size, max_a_seq_len]
        history_m: (history_m_time, history_m_type), tuple
        history_m_time&history_m_type: tensor with shape [batch_size, max_m_seq_len]
        return conditional-intensity-function for each a in history_a
        """
        # todo: 先将batch_tensor: history_a & history_m 转换成原始的 生成数据那时候用的形式(data[sample_ID][predicate_idx]['time']: [time stampls list])，然后根据公式计算出这些batch 的 a 的
        #  likelihood，avg on batch_size，得到当前batch下的likelihood
    def get_temporal_feature(self, cur_time, head_idx, history, cur_logic_rule_template):
        feature = 0
        occur_time_dict = dict()
        for idx, body_predicate_idx in enumerate(cur_logic_rule_template["body_predicate_idx"]):
            occur_time = np.array(history[body_predicate_idx]['time'])
            mask = (occur_time <= cur_time)  # find corresponding history
            # mask: filter all the history time points that satisfies the conditions which will boost the head predicate
            occur_time_dict[body_predicate_idx] = occur_time[mask]
        occur_time_dict[head_idx] = [cur_time]

        # compute features whenever any item of the transition_item_dic is nonempty
        history_transition_len = [len(i) for i in occur_time_dict.values()]
        if min(history_transition_len) > 0:
            # need to compute feature using logic rules
            time_combination = np.array(list(itertools.product(*occur_time_dict.values())))
            # get all possible time combinations
            time_combination_dict = {}
            for i, idx in enumerate(list(occur_time_dict.keys())):
                time_combination_dict[idx] = time_combination[:, i]
            temporal_kernel = np.ones(len(time_combination))
            for idx, temporal_relation_idx in enumerate(cur_logic_rule_template['temporal_relation_idx']):
                time_difference = time_combination_dict[temporal_relation_idx[0]] - time_combination_dict[temporal_relation_idx[1]]
                if cur_logic_rule_template['temporal_relation_type'][idx] == 'BEFORE':
                    temporal_kernel *= (time_difference < -self.Time_tolerance) * \
                                       np.exp(-self.decay_rate * (cur_time - time_combination_dict[temporal_relation_idx[0]]))
                if cur_logic_rule_template['temporal_relation_type'][idx] == 'EQUAL':
                    temporal_kernel *= (abs(time_difference) <= self.Time_tolerance) * \
                                       np.exp(-self.decay_rate * (cur_time - time_combination_dict[temporal_relation_idx[0]]))
                if cur_logic_rule_template['temporal_relation_type'][idx] == 'AFTER':
                    temporal_kernel *= (time_difference > self.Time_tolerance) * \
                                       np.exp(-self.decay_rate * (cur_time - time_combination_dict[temporal_relation_idx[1]]))
            feature = np.sum(temporal_kernel)
        return feature  # a scalar

    def get_formula_effect(self, cur_logic_rule_template):
        if cur_logic_rule_template['head_predicate_sign'][0] == 1:
            formula_effect = 1
        else:
            formula_effect = -1
        return formula_effect

    def intensity(self, cur_time, head_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []
        for f_idx in list(self.logic_template[head_idx].keys()):
            cur_weight = self.model_parameters[head_idx][f_idx]['weight']
            weight_formula.append(cur_weight)

            cur_feature = self.get_temporal_feature(cur_time, head_idx, history, self.logic_template[head_idx][f_idx])
            feature_formula.append(torch.tensor([cur_feature], dtype=torch.float64))

            cur_effect = self.get_formula_effect(self.logic_template[head_idx][f_idx])
            effect_formula.append(torch.tensor([cur_effect], dtype=torch.float64))

        intensity = torch.cat(weight_formula, dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0)
        intensity = self.model_parameter[head_idx]['base'] + torch.sum(intensity)
        if intensity.item() >= 0:
            if intensity.item() >= self.model_parameter[head_idx]['base'].item():
                final_intensity = intensity
            else:
                final_intensity = self.model_parameter[head_idx]['base']
            # final_intensity = torch.max(torch.tensor([intensity.item(), self.model_parameter[head_predicate_idx]['base']]))
            return final_intensity
        else:
            return torch.exp(intensity)

    def intensity_log_sum(self, head_predicate_idx, data_sample):
        pass
    def intensity_integral(self, head_predicate_idx, data_sample, T_max):
        pass
    def log_likelihood(self, dataset, sample_ID_batch, T_max):
        pass

    # todo: above will compute the sum of log_lh over batched data, average operation over scalar:batch_size
    #  to be the final obj func optimized will appear in the outside train_epoch in main.py























