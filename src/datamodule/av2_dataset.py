from typing import List
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = None,
        num_historical_steps: int = 50,
        sequence_origins: List[int] = [50],
        radius: float = 150.0,
        train_mode: str = 'only_focal',

    ):
        assert sequence_origins[-1] == 50 and num_historical_steps <= 50
        assert train_mode in ['only_focal', 'focal_and_scored']
        assert split in ['train', 'val', 'test']
        super(Av2Dataset, self).__init__()
        
        self.data_folder = Path(data_root) / split
        self.file_list = sorted(list(self.data_folder.glob('*.pt')))
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = 0 if split =='test' else 60
        self.sequence_origins = sequence_origins
        self.mode = 'only_focal' if split != 'train' else train_mode
        self.radius = radius

        print(
            f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}'
        )
        
        self[0]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index])
        data = self.process(data)
        return data
    
    # x_positions torch.Size([56, 110, 2])
    # x_attr torch.Size([56, 3])
    # x_angles torch.Size([56, 110])
    # x_velocity torch.Size([56, 110])
    # x_valid_mask torch.Size([56, 110])
    # lane_positions torch.Size([25, 20, 2])
    # lane_attr torch.Size([25, 3])
    # is_intersections torch.Size([25])
    # 'scenario_id': str
    # agent_ids: list, [56]
    # focal_idx: [5]
    # scored_idx: [0, 10]
    # city: str

    def process(self, data):
        sequence_data = []
        train_idx = [data['focal_idx']]
        
        # 'only_focal' for single-agent setting, 'focal_and_scored' for multi-agent setting
        if self.mode == 'focal_and_scored': # 'only_focal'
            train_idx += data['scored_idx']
        
        for cur_step in self.sequence_origins: #[30, 40, 50]
            for ag_idx in train_idx:
                ag_dict = self.process_single_agent(data, ag_idx, cur_step)
            sequence_data.append(ag_dict) # list. [3]
            
        # sequence_data[0]['target'] - [12, 60, 2]
        # sequence_data[1]['target'] - [15, 60, 2]
        # sequence_data[2]['target'] - [22, 60, 2]
        
        return sequence_data

    def process_single_agent(self, data, idx, step=50):
        # info for cur_agent on cur_step
        cur_agent_id = data['agent_ids'][idx] # '51241'
        origin = data['x_positions'][idx, step - 1].double() #[2]
        theta = data['x_angles'][idx, step - 1].double()
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        ) #[2, 2]
        
        # Within radius at current timestep
        ag_mask = torch.norm(data['x_positions'][:, step - 1] - origin, dim=-1) < self.radius # [56]
        ag_mask = ag_mask * data['x_valid_mask'][:, step - 1] # [56]
        ag_mask[idx] = False # False means invalid
        

        # transform agents to local
        st, ed = step - self.num_historical_steps, step + self.num_future_steps # 0, 90,  self.num_historical_steps=30, self.num_future_steps=60
        attr = torch.cat([data['x_attr'][[idx]], data['x_attr'][ag_mask]]) # [1, 3] + [12, 3] = [13, 3]
        pos = data['x_positions'][:, st: ed] # [56, 110, 2] -> [56, 90, 2]
        pos = torch.cat([pos[[idx]], pos[ag_mask]]) # [13, 90, 2]
        head = data['x_angles'][:, st: ed] # [56, 110] -> [56, 90]
        head = torch.cat([head[[idx]], head[ag_mask]]) # [13, 90]
        vel = data['x_velocity'][:, st: ed] # [56, 110] -> [56, 90]
        vel = torch.cat([vel[[idx]], vel[ag_mask]]) # [13, 90]
        valid_mask = data['x_valid_mask'][:, st: ed] # [56, 110] -> [56, 90]
        valid_mask = torch.cat([valid_mask[[idx]], valid_mask[ag_mask]]) # [13, 90]

        pos[valid_mask] = torch.matmul(pos[valid_mask].double() - origin, rotate_mat).to(torch.float32) # [13, 90, 2]
        head[valid_mask] = (head[valid_mask] - theta + np.pi) % (2 * np.pi) - np.pi # [13, 90]

        # transform lanes to local
        l_pos = data['lane_positions'] # [25, 20, 2]
        l_attr = data['lane_attr'] # [25, 3]
        l_is_int = data['is_intersections'] # [25]
        l_pos = torch.matmul(l_pos.reshape(-1, 2).double() - origin, rotate_mat).reshape(-1, l_pos.size(1), 2).to(torch.float32) # [25, 20, 2]
        
        l_ctr = l_pos[:, 9:11].mean(dim=1) # [25, 2]
        l_head = torch.atan2(
            l_pos[:, 10, 1] - l_pos[:, 9, 1],
            l_pos[:, 10, 0] - l_pos[:, 9, 0],
        ) # [25]
        l_valid_mask = (
            (l_pos[:, :, 0] > -self.radius) & (l_pos[:, :, 0] < self.radius)
            & (l_pos[:, :, 1] > -self.radius) & (l_pos[:, :, 1] < self.radius)
        ) # [25, 20]

        l_mask = l_valid_mask.any(dim=-1) #[25]
        l_pos = l_pos[l_mask] # [25, 20, 2]
        l_is_int = l_is_int[l_mask] # [25]
        l_attr = l_attr[l_mask] # [25, 3]
        l_ctr = l_ctr[l_mask] # [25, 2]
        l_head = l_head[l_mask] # [25]
        l_valid_mask = l_valid_mask[l_mask] # [25, 20]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        ) # [25, 20, 2]

        # remove outliers
        nearest_dist = torch.cdist(pos[:, self.num_historical_steps - 1, :2],
                                   l_pos.view(-1, 2)).min(dim=1).values # [13, 2], [25x20, 2] -> [13, 500] -> [13], agent pos to lane minimum distance
        ag_mask = nearest_dist < 5 #[13]
        ag_mask[0] = True
        pos = pos[ag_mask] # [12, 90, 2]
        head = head[ag_mask] # [12, 90]
        vel = vel[ag_mask] # [12, 90]
        attr = attr[ag_mask] # [12, 3]
        valid_mask = valid_mask[ag_mask] # [12, 90]

        # post_process
        head = head[:, :self.num_historical_steps] # [12, 30]
        vel_future = vel[:, self.num_historical_steps:] # [12, 60]
        vel = vel[:, :self.num_historical_steps] # [12, 30]
        pos_ctr = pos[:, self.num_historical_steps - 1].clone() # [12, 2]
        if self.num_future_steps > 0:
            type_mask = attr[:, [-1]] != 3 # [12, 1]
            pos, target = pos[:, :self.num_historical_steps], pos[:, self.num_historical_steps:] # [12, 30, 2], [12, 60, 2]
            target_mask = type_mask & valid_mask[:, [self.num_historical_steps - 1]] & valid_mask[:, self.num_historical_steps:] # [12, 60]
            valid_mask = valid_mask[:, :self.num_historical_steps] # [12, 30]
            target = torch.where(
                target_mask.unsqueeze(-1),
                target - pos_ctr.unsqueeze(1), torch.zeros(pos_ctr.size(0), 60, 2),   
            ) # [12, 60, 2]
        else:
            target = target_mask = None

        diff_mask = valid_mask[:, :self.num_historical_steps - 1] & valid_mask[:, 1: self.num_historical_steps] # [12, 29]
        tmp_pos = pos.clone() # [12, 30, 2]
        pos_diff = pos[:, 1:self.num_historical_steps] - pos[:, :self.num_historical_steps - 1] # [12, 29, 2]
        
        # add target velocity and acceleration 
        target_diff = None
        if target is not None:
            target_diff_tmp = torch.cat((pos[:, -1].unsqueeze(1), target), dim=1) # [12, 61, 2] TODO: pos and target in the same coordinate?
            target_diff = target_diff_tmp[:, 1:self.num_future_steps+1] - target_diff_tmp[:, :self.num_future_steps] # [12, 60, 2]
            target_diff_tmp = target_diff.clone() # [12, 60, 2]
            diff_mask_target_tmp = torch.cat((valid_mask[:,-1].unsqueeze(1), target_mask), dim=1) # [12, 61]
            diff_mask_target = diff_mask_target_tmp[:, 1:self.num_future_steps + 1] & diff_mask_target_tmp[:, : self.num_future_steps] # [12, 60]
            target_diff[:, :] = torch.where(
                diff_mask_target.unsqueeze(-1),
                target_diff_tmp, torch.zeros(target_diff_tmp.size(0), self.num_future_steps, 2)
            ) # [12, 60, 2]
        
        pos[:, 1:self.num_historical_steps] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff, torch.zeros(pos.size(0), self.num_historical_steps - 1, 2)
        ) # ([12, 30, 2]
        pos[:, 0] = torch.zeros(pos.size(0), 2) # [12, 30, 2]

        tmp_vel = vel.clone() # [12, 30]
        vel_diff = vel[:, 1:self.num_historical_steps] - vel[:, :self.num_historical_steps - 1] # [12, 29]
        vel[:, 1:self.num_historical_steps] = torch.where(
            diff_mask,
            vel_diff, torch.zeros(vel.size(0), self.num_historical_steps - 1)
        ) # [12, 30]
        vel[:, 0] = torch.zeros(vel.size(0)) # [12, 30]
        
        # add target velocity and acceleration
        if target is not None:
            tmpvel_future = vel_future.clone() # [12, 60]
            tmpvel_future = torch.cat((tmp_vel[:, -1].unsqueeze(1), tmpvel_future), dim=1) # [12, 61]
            vel_diff_future= tmpvel_future[:, 1:self.num_future_steps+1] - tmpvel_future[:, :self.num_future_steps] # [12, 60]
            vel_future[:, :] = torch.where(
                diff_mask_target,
                vel_diff_future, torch.zeros(vel_diff_future.size(0), self.num_future_steps)
            ) # [12, 60]
        
        return {
            'target': target,
            'target_diff': target_diff,
            'target_vel_diff': vel_future,
            'target_mask': target_mask,

            'x_positions_diff': pos,
            'x_positions': tmp_pos,
            'x_attr': attr,
            'x_centers': pos_ctr,
            'x_angles': head,
            'x_velocity': tmp_vel,
            'x_velocity_diff': vel,
            'x_valid_mask': valid_mask,

            'lane_positions': l_pos,
            'lane_centers': l_ctr,
            'lane_angles': l_head,
            'lane_attr': l_attr,
            'lane_valid_mask': l_valid_mask,
            'is_intersections': l_is_int,
            
            'origin': origin.view(1, 2),
            'theta': theta.view(1),
            'scenario_id': data['scenario_id'],
            'track_id': cur_agent_id,
            'city': data['city'],
            'timestamp': torch.Tensor([step * 0.1])
        }
    

def collate_fn(seq_batch):
    seq_data = []
    for i in range(len(seq_batch[0])):
        batch = [b[i] for b in seq_batch]
        data = {}

        for key in [
            'x_positions_diff',
            'x_attr',
            'x_positions',
            'x_centers',
            'x_angles',
            'x_velocity',
            'x_velocity_diff',
            'lane_positions',
            'lane_centers',
            'lane_angles',
            'lane_attr',
            'is_intersections',
        ]:
            data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

        if 'x_scored' in batch[0]:
            data['x_scored'] = pad_sequence(
                [b['x_scored'] for b in batch], batch_first=True
            )

        if batch[0]['target'] is not None:
            data['target'] = pad_sequence([b['target'] for b in batch], batch_first=True)
            data['target_diff'] = pad_sequence([b['target_diff'] for b in batch], batch_first=True)
            data['target_vel_diff'] = pad_sequence([b['target_vel_diff'] for b in batch], batch_first=True)
            data['target_mask'] = pad_sequence(
                [b['target_mask'] for b in batch], batch_first=True, padding_value=False
            )

        for key in ['x_valid_mask', 'lane_valid_mask']:
            data[key] = pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=False
            )

        data['x_key_valid_mask'] = data['x_valid_mask'].any(-1)
        data['lane_key_valid_mask'] = data['lane_valid_mask'].any(-1)

        data['scenario_id'] = [b['scenario_id'] for b in batch]
        data['track_id'] = [b['track_id'] for b in batch]

        data['origin'] = torch.cat([b['origin'] for b in batch], dim=0)
        data['theta'] = torch.cat([b['theta'] for b in batch])
        data['timestamp'] = torch.cat([b['timestamp'] for b in batch])
        seq_data.append(data)
        

    # print(len(seq_data), len(seq_batch), len(seq_batch[0])) 3, 16, 3
    return seq_data
