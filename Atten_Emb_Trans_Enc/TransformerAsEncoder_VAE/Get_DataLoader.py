from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import partial
import torch


class SynDataset(Dataset):
    def __init__(self, data, action_type_list, mental_type_list):
        self.data = data  # data has form: data[sample_ID][predicate_idx]['time'] = [...]
        self.action_type_list = action_type_list  # cur: [1, 2]
        self.mental_type_list = mental_type_list  # cur: [0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_sample_data = self.data[idx]
        action_time_list = {}  # {a1: [t1, t2, ...], a2: [t1, t2, ...] }
        for a_id in self.action_type_list:
            action_time_list[a_id] = cur_sample_data[a_id]["time"]
        mental_time_list = {}   # {m0: [t1, t2, ...]}
        for m_id in self.mental_type_list:
            mental_time_list[m_id] = cur_sample_data[m_id]["time"]
        return action_time_list, mental_time_list


def collate_fn(batch_samples, a_type_list, m_type_list):

    org_a_batch, org_m_batch = list(zip(*batch_samples))
    batch_size = len(org_a_batch)
    all_a_org_seq = []
    all_m_org_seq = []
    for a in org_a_batch:
        cur_sample_a_seq = []
        for a_type in a_type_list:
            cur_sample_a_seq += a[a_type]
        all_a_org_seq += [cur_sample_a_seq]

    for m in org_m_batch:
        cur_sample_m_seq = []
        for m_type in m_type_list:
            cur_sample_m_seq += m[m_type]
        all_m_org_seq += [cur_sample_m_seq]
    max_a_seq_len = max(len(a_org_seq) for a_org_seq in all_a_org_seq)
    max_m_seq_len = max(len(m_org_seq) for m_org_seq in all_m_org_seq)
    pad_a_type_batch = torch.zeros((batch_size, max_a_seq_len))
    pad_a_time_batch = torch.zeros((batch_size, max_a_seq_len))
    pad_m_type_batch = torch.zeros((batch_size, max_m_seq_len))
    pad_m_time_batch = torch.zeros((batch_size, max_m_seq_len))

    for b_id in range(batch_size):
        pad_a_seq_tuple_list = []
        pad_m_seq_tuple_list = []
        for a_type in a_type_list:
            pad_a_seq_tuple_list += [(a_type-1, time) for time in org_a_batch[b_id][a_type]]  # action_type_list: [2, 3]
        pad_a_seq_tuple_list.sort(key=lambda x: x[1])
        pad_a_type_batch[b_id] = torch.tensor([item[0] for item in pad_a_seq_tuple_list] + [0] * (pad_a_type_batch.shape[1] - len(pad_a_seq_tuple_list)))
        pad_a_time_batch[b_id] = torch.tensor([item[1] for item in pad_a_seq_tuple_list] + [0] * (pad_a_time_batch.shape[1] - len(pad_a_seq_tuple_list)))

        for m_type in m_type_list:
            pad_m_seq_tuple_list += [(m_type, time) for time in org_m_batch[b_id][m_type]]  # mental_type_list: [1]
        pad_m_seq_tuple_list.sort(key=lambda x: x[1])
        pad_m_type_batch[b_id] = torch.tensor([item[0] for item in pad_m_seq_tuple_list] + [0] * (pad_m_type_batch.shape[1] - len(pad_m_seq_tuple_list)))
        pad_m_time_batch[b_id] = torch.tensor([item[1] for item in pad_m_seq_tuple_list] + [0] * (pad_m_type_batch.shape[1] - len(pad_m_seq_tuple_list)))

    return pad_a_time_batch, pad_a_type_batch, pad_m_time_batch, pad_m_type_batch


def get_dataloader(data, action_type_list, mental_type_list, batch_size):
    dataset = SynDataset(data, action_type_list, mental_type_list)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=partial(collate_fn, a_type_list=action_type_list, m_type_list=mental_type_list),
                            num_workers=2)
    return dataloader