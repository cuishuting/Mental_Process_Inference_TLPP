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
        mental_time_list = {}
        for m_id in self.mental_type_list:
            mental_time_list[m_id] = cur_sample_data[m_id]["time"]

        return action_time_list, mental_time_list


def collate_fn(batch_samples, a_type_list, m_type_list):
    org_a_batch, org_m_batch = list(zip(*batch_samples))
    all_a_org_seq = []
    all_m_org_seq = []
    for a_type in a_type_list:
        all_a_org_seq += [a[a_type] for a in org_a_batch]
    for m_type in m_type_list:
        all_m_org_seq += [m[m_type] for m in org_m_batch]
    max_a_seq_len = max(len(a_org_seq) for a_org_seq in all_a_org_seq)
    max_m_seq_len = max(len(m_org_seq) for m_org_seq in all_m_org_seq)
    pad_a_seq = {}
    pad_m_seq = {}
    # todo: store each real occured time seq length for each action and mental predicate in each batch
    real_a_time_num = {}
    real_m_time_num = {}
    for a_type in a_type_list:
        pad_a_seq[a_type] = torch.tensor([a[a_type] + [0]*(max_a_seq_len - len(a[a_type])) for a in org_a_batch])
        real_a_time_num[a_type] = torch.tensor([len(a[a_type]) for a in org_a_batch])
    for m_type in m_type_list:
        pad_m_seq[m_type] = torch.tensor([m[m_type] + [0] * (max_m_seq_len - len(m[m_type])) for m in org_m_batch])
        real_m_time_num[m_type] = torch.tensor([len(m[m_type]) for m in org_m_batch])
    return pad_a_seq, pad_m_seq, real_a_time_num, real_m_time_num


def get_dataloader(data, action_type_list, mental_type_list, batch_size):
    dataset = SynDataset(data, action_type_list, mental_type_list)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=partial(collate_fn, a_type_list=action_type_list, m_type_list=mental_type_list),
                            num_workers=2)
    return dataloader











