import numpy as np


def get_m_occur_grids_list(m_pad_batch, time_horizon, sep_for_grids, m_type_list, real_m_time_num, batch_size):
    """
    m_pad_batch: dict with padded time seq, like: {0: torch.tensor([[1,3,0], [4,0,0], [3,4,5]])} each value has shape: [batch_size, max_seq_len]
    real_m_time_num:  like {0: tensor([16, 13, 20, 19, 11])}  with batch_size == 5
    output: True-False array with True signifying mental occurrence in cur grid, shape: [batch_size, num_mental_types, num_of_grids]
    """
    org_time_dict = {}
    for m in m_type_list:
        org_time_dict[m] = {}
        for b_id in range(batch_size):
            org_time_dict[m][b_id] = m_pad_batch[m][b_id][:real_m_time_num[m][b_id]]
    num_of_grids = int(time_horizon / sep_for_grids)
    processed_data = np.zeros([batch_size, len(m_type_list), num_of_grids])
    for m_id, m in enumerate(m_type_list):
        for b_id in range(batch_size):
            cur_real_time_seq = org_time_dict[m][b_id]
            cur_check_time_pos = 0
            for g_id in range(num_of_grids):
                cur_grid_right_time = (g_id + 1) * sep_for_grids
                if (cur_check_time_pos < len(cur_real_time_seq)) and cur_real_time_seq[cur_check_time_pos] <= cur_grid_right_time:
                    processed_data[b_id][m_id][g_id] = cur_real_time_seq[cur_check_time_pos]
                    cur_check_time_pos += 1
                else:
                    continue

    return processed_data != 0, org_time_dict
