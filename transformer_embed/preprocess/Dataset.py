import numpy as np
import torch
import torch.utils.data

from transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        
        self.mental_predicate_set = Constants.mental_predicate_set
        self.action_predicate_set = Constants.action_predicate_set
        self.head_predicate_set = Constants.head_predicate_set
        self.total_predicate_set = Constants.total_predicate_set
        
        
        self.mental_time = []
        self.mental_type = []
        self.action_time = []
        self.action_type = []
        for inst_idx in range(len(data)):
            self.mental_time.append([])
            self.mental_type.append([])
            self.action_time.append([])
            self.action_type.append([])
            for elem in data[inst_idx]:
                ## there is no event type 0, which means that the event type start from 1, so we can use 0 as padding
                if elem['type_event'] in self.mental_predicate_set:
                    self.mental_time[inst_idx].append(elem['time_since_start'])
                    self.mental_type[inst_idx].append(elem['type_event'])
                elif elem['type_event'] in self.action_predicate_set:
                    self.action_time[inst_idx].append(elem['time_since_start'])
                    self.action_type[inst_idx].append(elem['type_event'])

        self.length = len(data) ## for data_hawkes/train.pkl, there are 8000 seqs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.mental_time[idx], self.mental_type[idx], self.action_time[idx], self.action_type[idx]


def padding(insts_time, insts_type):
    """ Pad the instance to the the number of grid in batch, since we discretize the timeline. """
    
    num_grid = int(Constants.time_horizon/Constants.grid_length)
    
    discrete_time_batch_seq = []
    discrete_type_batch_seq = []
    for inst_idx in range(len(insts_time)):
        discrete_time_batch_seq.append([])
        discrete_type_batch_seq.append([])
        for grid_idx in range(num_grid):
            grid_lower_bound = grid_idx * Constants.grid_length
            grid_upper_bound = (grid_idx + 1) * Constants.grid_length
            
            event_time_list = []
            event_type_list = []
            for event_idx in range(len(insts_time[inst_idx])):
                if insts_time[inst_idx][event_idx] >= grid_lower_bound and insts_time[inst_idx][event_idx] < grid_upper_bound:
                    event_time_list.append(insts_time[inst_idx][event_idx])
                    event_type_list.append(insts_type[inst_idx][event_idx])
                else:
                    continue
            if len(event_time_list) > 0: ##### 某个grid上至少有一个event发生
                discrete_time_batch_seq[inst_idx].append(event_time_list[0]) ##### 选择这个grid上的第一个event
                discrete_type_batch_seq[inst_idx].append(event_type_list[0]) ##### 选择这个grid上的第一个event
            else:
                discrete_time_batch_seq[inst_idx].append(Constants.PAD) ##### 在该grid所在的位置补0
                discrete_type_batch_seq[inst_idx].append(Constants.PAD) ##### 在该grid所在的位置补0
    
    return torch.tensor(discrete_time_batch_seq, dtype=torch.float32), torch.tensor(discrete_type_batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    mental_time, mental_type, action_time, action_type = list(zip(*insts))

    mental_time, mental_type = padding(mental_time, mental_type)
    action_time, action_type = padding(action_time, action_type)

    return mental_time, mental_type, action_time, action_type


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
