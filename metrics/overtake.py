import numpy as np
import pandas as pd

class OvertakeSimulator:

    def __init__(self):
        self.drag_coeff_base = 1.0
        self.drs_drag_reduction = 0.25
        self.slipstream_coeff = 0.08
        self.air_density = 1.225
        self.frontal_area = 1.6
        self.power_base_hp = 1000
        self.ers_boost_hp = 160
        self.mass_kg = 798

    def calculate_acceleration(self, velocity, drs_active, slipstream_dist, ers_active):
        drag_c = self.drag_coeff_base
        if drs_active:
            drag_c -= self.drs_drag_reduction
        if slipstream_dist < 60:
            slipstream_effect = self.slipstream_coeff * (1 - slipstream_dist / 60)
            drag_c -= slipstream_effect
        drag_force = 0.5 * self.air_density * self.frontal_area * drag_c * velocity ** 2
        power_hp = self.power_base_hp + (self.ers_boost_hp if ers_active else 0)
        power_watts = power_hp * 745.7
        force_engine = power_watts / max(velocity, 0.1)
        net_force = force_engine - drag_force
        accel = net_force / self.mass_kg
        return accel

    def simulate_straight(self, length_m, exit_speed_mps, tyre_delta_sec):
        dt = 0.05
        pos_chase = 0
        pos_defend = 45
        vel_chase = exit_speed_mps + tyre_delta_sec * 1.5
        vel_defend = exit_speed_mps
        time_elapsed = 0
        overtake_occured = False
        overtake_time = None
        while pos_defend < length_m:
            acc_chase = self.calculate_acceleration(vel_chase, drs_active=True, slipstream_dist=pos_defend - pos_chase, ers_active=True)
            acc_defend = self.calculate_acceleration(vel_defend, drs_active=False, slipstream_dist=999, ers_active=False)
            vel_chase += acc_chase * dt
            vel_defend += acc_defend * dt
            pos_chase += vel_chase * dt
            pos_defend += vel_defend * dt
            time_elapsed += dt
            if pos_chase > pos_defend and (not overtake_occured):
                overtake_occured = True
                overtake_time = time_elapsed
        return {'overtake': overtake_occured, 'time': overtake_time, 'margin_m': pos_chase - pos_defend}

def run_overtake_simulation_grid():
    sim = OvertakeSimulator()
    straight_lengths = np.linspace(400, 1200, 40)
    tyre_deltas = np.linspace(0, 2.0, 40)
    results = []
    for length in straight_lengths:
        for delta in tyre_deltas:
            res = sim.simulate_straight(length, exit_speed_mps=60, tyre_delta_sec=delta)
            margin = res['margin_m']
            prob = 1 / (1 + np.exp(-0.15 * margin))
            results.append({'StraightLength': length, 'TyreDelta': delta, 'OvertakeProb': prob, 'Margin': margin})
    return pd.DataFrame(results)