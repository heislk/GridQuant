import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TyreModel:
    base_deg_per_lap: float
    volatility: float
    cliff_lap: int
    cliff_penalty: float

@dataclass
class Strategy:
    name: str
    color: str
    compounds: List[str]
    pit_laps: List[int]

class RaceSimulator:

    def __init__(self, total_laps: int, base_lap_time: float, pit_loss: float):
        self.total_laps = total_laps
        self.base_lap_time = base_lap_time
        self.pit_loss = pit_loss
        self.pit_loss_std = 1.5
        self.sc_probability = 0.008
        self.sc_delay = 25.0
        self.traffic_probability = 0.15
        self.traffic_penalty_mean = 1.2
        self.mistake_probability = 0.002
        self.mistake_penalty_mean = 3.0
        self.tyre_models = {'SOFT': TyreModel(base_deg_per_lap=0.065, volatility=0.15, cliff_lap=22, cliff_penalty=0.25), 'MEDIUM': TyreModel(base_deg_per_lap=0.04, volatility=0.12, cliff_lap=32, cliff_penalty=0.2), 'HARD': TyreModel(base_deg_per_lap=0.025, volatility=0.1, cliff_lap=45, cliff_penalty=0.15)}
        self.compound_offset = {'SOFT': 0.0, 'MEDIUM': 0.6, 'HARD': 1.2}

    def get_lap_time(self, compound: str, tyre_age: int, lap_number: int, rng: np.random.Generator, in_traffic: bool) -> float:
        model = self.tyre_models[compound]
        fuel_factor = (self.total_laps - lap_number) / self.total_laps * 0.035 * self.total_laps
        degradation = model.base_deg_per_lap * tyre_age
        if tyre_age > model.cliff_lap:
            degradation += model.cliff_penalty * (tyre_age - model.cliff_lap)
        compound_penalty = self.compound_offset[compound]
        noise = rng.normal(0, model.volatility)
        
        traffic_loss = 0.0
        if in_traffic:
            traffic_loss = rng.exponential(self.traffic_penalty_mean)
            
        mistake_loss = 0.0
        if rng.random() < self.mistake_probability:
            mistake_loss = rng.exponential(self.mistake_penalty_mean)

        lap_time = self.base_lap_time + fuel_factor + degradation + compound_penalty + noise + traffic_loss + mistake_loss
        return lap_time

    def simulate_race(self, strategy: Strategy, rng: np.random.Generator) -> Dict:
        total_time = 0.0
        lap_times = []
        tyre_ages = []
        compounds_used = []
        current_compound_idx = 0
        current_compound = strategy.compounds[0]
        tyre_age = 0
        in_traffic = False
        traffic_counter = 0

        for lap in range(1, self.total_laps + 1):
            if current_compound_idx < len(strategy.pit_laps) and lap == strategy.pit_laps[current_compound_idx]:
                pit_time = self.pit_loss + rng.normal(0, self.pit_loss_std)
                total_time += pit_time
                current_compound_idx += 1
                current_compound = strategy.compounds[current_compound_idx]
                tyre_age = 0
                
                if rng.random() < self.traffic_probability:
                    in_traffic = True
                    traffic_counter = rng.integers(2, 6)

            if in_traffic:
                traffic_counter -= 1
                if traffic_counter <= 0:
                    in_traffic = False

            tyre_age += 1
            lap_time = self.get_lap_time(current_compound, tyre_age, lap, rng, in_traffic)
            
            if rng.random() < self.sc_probability:
                lap_time += self.sc_delay
                
            lap_times.append(lap_time)
            tyre_ages.append(tyre_age)
            compounds_used.append(current_compound)
            total_time += lap_time
        return {'total_time': total_time, 'lap_times': lap_times, 'tyre_ages': tyre_ages, 'compounds': compounds_used}

    def run_monte_carlo(self, strategy: Strategy, n_sims: int=10000, seed: int=42) -> Dict:
        rng = np.random.default_rng(seed)
        total_times = np.zeros(n_sims)
        all_lap_times = np.zeros((n_sims, self.total_laps))
        for i in range(n_sims):
            result = self.simulate_race(strategy, rng)
            total_times[i] = result['total_time']
            all_lap_times[i] = result['lap_times']
        mean_lap_times = np.mean(all_lap_times, axis=0)
        std_lap_times = np.std(all_lap_times, axis=0)
        return {'strategy': strategy.name, 'total_times': total_times, 'mean': np.mean(total_times), 'std': np.std(total_times), 'median': np.median(total_times), 'var_5': np.percentile(total_times, 5), 'var_95': np.percentile(total_times, 95), 'best': np.min(total_times), 'worst': np.max(total_times), 'mean_lap_times': mean_lap_times, 'std_lap_times': std_lap_times}

def compare_strategies(simulator: RaceSimulator, strategies: List[Strategy], n_sims: int=10000) -> Dict:
    results = {}
    for strategy in strategies:
        print(f'  Simulating: {strategy.name}')
        results[strategy.name] = simulator.run_monte_carlo(strategy, n_sims)
    comparisons = {}
    names = list(results.keys())
    for (i, name_a) in enumerate(names):
        for name_b in names[i + 1:]:
            times_a = results[name_a]['total_times']
            times_b = results[name_b]['total_times']
            wins_a = np.sum(times_a < times_b)
            win_rate_a = wins_a / n_sims
            mean_delta = results[name_a]['mean'] - results[name_b]['mean']
            comparisons[f'{name_a} vs {name_b}'] = {'win_rate_a': win_rate_a, 'win_rate_b': 1 - win_rate_a, 'mean_delta': mean_delta}
    return {'results': results, 'comparisons': comparisons, 'strategies': strategies}

def get_default_strategies() -> List[Strategy]:
    return [Strategy('Soft-Medium 1-Stop', '#ff6b6b', ['SOFT', 'MEDIUM'], [20]), Strategy('Soft-Hard 1-Stop', '#4ecdc4', ['SOFT', 'HARD'], [15]), Strategy('Medium-Hard 1-Stop', '#ffe66d', ['MEDIUM', 'HARD'], [25]), Strategy('Soft-Med-Soft 2-Stop', '#a29bfe', ['SOFT', 'MEDIUM', 'SOFT'], [15, 35])]