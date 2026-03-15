import numpy as np
from typing import List, Dict, Tuple
from .simulation import TyreModel

class StrategyOptimizer:

    def __init__(self, total_laps: int, pit_loss: float, tyre_models: Dict[str, TyreModel], base_lap_time: float):
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        self.tyre_models = tyre_models
        self.base_lap_time = base_lap_time
        self.compound_offset = {'SOFT': 0.0, 'MEDIUM': 0.6, 'HARD': 1.2}

    def get_lap_cost(self, compound: str, age: int) -> float:
        model = self.tyre_models[compound]
        deg = model.base_deg_per_lap * age
        if age > model.cliff_lap:
            deg += model.cliff_penalty * (age - model.cliff_lap)
        return self.base_lap_time + self.compound_offset[compound] + deg

    def solve(self) -> Dict:
        dp = {}
        parent = {}
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        for comp in compounds:
            dp[0, comp, 0] = 0.0
            parent[0, comp, 0] = None
        for lap in range(self.total_laps):
            current_states = [s for s in dp.keys() if s[0] == lap]
            for state in current_states:
                (curr_lap, curr_comp, curr_age) = state
                curr_time = dp[state]
                next_age = curr_age + 1
                if next_age < 60:
                    drive_cost = self.get_lap_cost(curr_comp, next_age)
                    next_state = (curr_lap + 1, curr_comp, next_age)
                    if next_state not in dp or curr_time + drive_cost < dp[next_state]:
                        dp[next_state] = curr_time + drive_cost
                        parent[next_state] = (*state, 'drive')
                next_age = curr_age + 1
                cost_stay = self.get_lap_cost(curr_comp, next_age)
                state_stay = (curr_lap + 1, curr_comp, next_age)
                if state_stay not in dp or curr_time + cost_stay < dp[state_stay]:
                    dp[state_stay] = curr_time + cost_stay
                    parent[state_stay] = (*state, 'stay')
                for new_comp in compounds:
                    if new_comp == curr_comp:
                        continue
                    cost_box = self.pit_loss + self.get_lap_cost(new_comp, 1)
                    state_box = (curr_lap + 1, new_comp, 1)
                    if state_box not in dp or curr_time + cost_box < dp[state_box]:
                        dp[state_box] = curr_time + cost_box
                        parent[state_box] = (*state, 'box_' + new_comp)
        end_states = [s for s in dp.keys() if s[0] == self.total_laps]
        best_end_state = min(end_states, key=lambda s: dp[s])
        min_total_time = dp[best_end_state]
        path = []
        curr = best_end_state
        while curr[0] > 0:
            (prev_lap, prev_comp, prev_age, action) = parent[curr]
            path.append({'lap': curr[0], 'compound': curr[1], 'action': action, 'time': dp[curr]})
            curr = (prev_lap, prev_comp, prev_age)
        path.reverse()
        return {'optimal_time': min_total_time, 'strategy': path}