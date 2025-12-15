from typing import List

class PerfConfig:
    def __init__(self, cat_cols: List[str], num_cols: List[str], target_col: str):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
