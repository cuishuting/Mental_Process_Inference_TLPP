import numpy as np
import torch
import torch.utils.data
import json
import constant


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """

        # print(self.label_type)
        self.length = len(data)
        self.action_dataset = []
        self.mental_dataset = []
        self.ora_action = []
        for keys, values in data.items():
            action_data = []
            mental_data = []
            ora_action = {}
            for key, value in values.items():
                if key in constant.mental_predicate_set:
                    if len(value['time']) > 0:
                        mental_data.append(value['time'])
                    else:
                        mental_data.append([constant.PAD])
                else:
                    if len(value['time']) > 0:
                        action_data.append(value['time'])
                    else:
                        action_data.append([constant.PAD])
                    ora_action[key] = value
            self.ora_action.append(ora_action)
            self.mental_dataset.append(mental_data)
            self.action_dataset.append(action_data)
            # new_dict = {str(key): value for key, value in value.items()}
            # self.train_data.append(json.dumps(new_dict))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.action_dataset[idx], self.mental_dataset[idx], self.ora_action[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = 0
    for inst in insts:
        for i in inst:
            max_len = max(max_len, len(i))
    new_lists = []
    for inst in insts:
        new_list = []
        for i in inst:
            new_list.append(i + [constant.PAD] * (max_len - len(i)))
        new_lists.append(new_list)
    batch_seq =  np.array(new_lists)
    
    return torch.tensor(batch_seq, dtype=torch.float32)

def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    action_dataset, mental_dataset, ora_action = list(zip(*insts))
    action_dataset = pad_time(action_dataset)
    mental_dataset = pad_time(mental_dataset)

    return [action_dataset, mental_dataset, ora_action]

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
