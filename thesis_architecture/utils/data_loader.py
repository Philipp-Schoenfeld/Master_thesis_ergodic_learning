import torch
import sqlite3
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class ErgodicTrajectoryDataset(Dataset):
    def __init__(self, db_path, table_name="runs"):
        conn = sqlite3.connect(db_path)
        # Jetzt laden wir auch x0_x und x0_y aus der Datenbank
        df = pd.read_sql_query(f"SELECT x0_x, x0_y, trajectory, shape FROM {table_name}", conn)
        conn.close()
        
        self.trajectories, self.contexts = self._parse_data(df)

    def _parse_data(self, df):
        parsed_trajs = []
        parsed_contexts = []
        
        for x, y, traj_blob, shape_str in zip(df['x0_x'], df['x0_y'], df['trajectory'], df['shape']):
            # 1. Trajektorie parsen
            shape_tuple = tuple(map(int, shape_str.split(',')))
            traj_array = np.frombuffer(traj_blob, dtype=np.float32).reshape(shape_tuple)
            parsed_trajs.append(traj_array[:, :2])
            
            # 2. Kontext (Startposition) speichern
            parsed_contexts.append([x, y])
            
        tensor_trajs = torch.tensor(np.array(parsed_trajs), dtype=torch.float32)
        tensor_contexts = torch.tensor(np.array(parsed_contexts), dtype=torch.float32)
        
        return tensor_trajs, tensor_contexts

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Gibt nun ein Tupel zurück: (Trajektorie, Kontextvektor)
        return self.trajectories[idx], self.contexts[idx]