"""
Scenario 2 (S2): Emergency Vehicle Platooning with Heterogeneous Dynamics
- Proposed Safe Predictor-Feedback CACC (Safe PF-CACC)
- Features: Extreme deceleration (-3.0 m/s^2) at high speeds (150 km/h),
  low V2X reliability (ETA = 0.5), individual time-headway limits (h_min),
  and Extended-case prediction formulation (D_PM,i > D_PM,i-1).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import expm
from collections import deque
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ==============================================================================
# [SECTION 1] SIMULATION CONFIGURATION
# ==============================================================================
class SimConfig:
    """Simulation environment variables and vehicle specifications."""

    # Time parameters
    DT = 0.01                # Time step (s)
    TOTAL_TIME = 80.0        # Total simulation time (s)
    WARMUP_TIME = 5.0        # Warm-up time for buffer initialization (s)

    # Visualization constants
    VEHICLE_LENGTH = 9.0     # Vehicle length for trucks/buses (m)
    VEHICLE_WIDTH = 2.5      # Vehicle width (m)
    LANE_WIDTH = 3.5         # Lane width (m)

    # Safe Reference Distance (S_ref) parameters
    S0 = 2.0                 # Minimum standstill distance (m)
    ETA = 0.5                # V2X communication reliability (0.5 = conservative)
    G = 9.81                 # Gravitational acceleration (m/s^2)

    # Low Pass Filter (LPF) time constant to handle extreme transients
    TAU_F = 2.0

    @staticmethod
    def get_leader_accel(t):
        """Leader vehicle acceleration profile (High-speed emergency braking)."""
        if t < 0: return 0.0
        if 0.0 <= t < 2.0: return 0.0      # Cruise at 25 m/s (90 km/h)
        if 2.0 <= t < 8.0: return 2.8      # Hard acceleration to ~150 km/h
        if 8.0 <= t < 20.0: return 0.0     # High-speed cruise
        if 20.0 <= t < 26.0: return -3.0   # Emergency extreme deceleration
        if 26.0 <= t < 38.0: return 0.0    # Low-speed cruise (~85 km/h)
        if 38.0 <= t < 44.0: return 2.5    # Re-acceleration to ~140 km/h
        if t >= 44.0: return 0.0           # Cruise
        return 0.0

    # Color palette for plotting (Leader + 4 Followers)
    COLORS = ['#1F449C', '#2CA02C', '#7B241C', '#F1948A', '#CB4335']

    # Heterogeneous vehicle specifications
    VEHICLE_SPECS = [
        # Leader (ID 0): Sedan
        {'id': 0, 'tau': 0.25, 'd_comm':0.00, 'd_proc':0.00, 'd_mech':0.00, 'mu':1.0, 'alpha':0, 'b':0, 'c':0, 'h_min':0.3},

        # Follower 1: SUV
        {'id': 1, 'tau': 0.30, 'd_comm':0.05, 'd_proc':0.03, 'd_mech':0.05, 'mu':1.0, 'alpha':4, 'b':8, 'c':2, 'h_min':0.49},

        # Follower 2: Heavy Truck (Extended case relative to SUV: longer tau, lower mu)
        {'id': 2, 'tau': 0.50, 'd_comm':0.05, 'd_proc':0.03, 'd_mech':0.05, 'mu':0.9, 'alpha':4, 'b':8, 'c':4, 'h_min':0.35},

        # Follower 3: Bus (Extended case relative to Truck due to d_mech mismatch)
        {'id': 3, 'tau': 0.40, 'd_comm':0.05, 'd_proc':0.03, 'd_mech':0.10, 'mu':1.0, 'alpha':8, 'b':10, 'c':3, 'h_min':0.35},

        # Follower 4: Truck (Degraded friction)
        {'id': 4, 'tau': 0.50, 'd_comm':0.05, 'd_proc':0.03, 'd_mech':0.05, 'mu':0.88, 'alpha':3, 'b':10, 'c':3, 'h_min':0.35},
    ]


# ==============================================================================
# [SECTION 2] SAFE REFERENCE DISTANCE LOGIC
# ==============================================================================
class SRefLogic:
    """Calculates the collision-free Safe Reference Distance (S_ref)."""

    @staticmethod
    def calculate_s_ref(v_ego, spec_ego, spec_front, D_i_ego):
        tau_ego = spec_ego['tau']
        d_comm = spec_ego['d_comm']
        mu_ego = spec_ego['mu']

        tau_front = spec_front['tau']
        d_proc_front = spec_front['d_proc']
        d_mech_front = spec_front['d_mech']
        mu_front = spec_front['mu']

        T_ego_total = d_comm + D_i_ego + tau_ego
        T_front_phys = d_proc_front + d_mech_front + tau_front
        delta_T_net = T_ego_total - T_front_phys

        # V2X prediction interval relies purely on processing/mechanical delay
        D_pred = D_i_ego

        inv_mu_ego = 1.0 / max(mu_ego, 0.1)
        inv_mu_front = 1.0 / max(mu_front, 0.1)
        delta_mu_inv = inv_mu_ego - inv_mu_front

        term_lag = v_ego * max(0.0, delta_T_net)
        term_pred = SimConfig.ETA * v_ego * D_pred
        term_brake = (v_ego**2 / (2 * SimConfig.G)) * max(0.0, delta_mu_inv)

        s_ref = SimConfig.S0 + term_lag - term_pred + term_brake

        return max(s_ref, SimConfig.S0)

    @staticmethod
    def calculate_h_t(s_ref, v_ego, h_min):
        """Calculates the effective time headway h(t) bounded by h_min."""
        safe_v = max(v_ego, 1.0)
        h_t = s_ref / safe_v
        return max(h_t, h_min)


# ==============================================================================
# [SECTION 3] VEHICLE DYNAMICS & CONTROL GAIN
# ==============================================================================
class VehicleDynamics:
    """Generates system matrices and control gains based on 3rd-order dynamics."""

    @staticmethod
    def get_system_matrices(tau_i, tau_pre):
        Gamma = np.zeros((5, 5))
        Gamma[0, 1] = -1; Gamma[0, 2] = 1
        Gamma[1, 3] = 1
        Gamma[2, 4] = 1
        Gamma[3, 3] = -1/tau_i
        Gamma[4, 4] = -1/tau_pre

        B_i = np.zeros(5); B_i[3] = 1/tau_i
        B_pre = np.zeros(5); B_pre[4] = 1/tau_pre
        return Gamma, B_i, B_pre

    @staticmethod
    def get_control_gain(h, tau, alpha, b, c):
        safe_h = max(h, 0.01)
        return np.array([
            (tau * alpha) / safe_h,
            -tau * (alpha + b),
            tau * b,
            -tau * c,
            tau * c
        ])


# ==============================================================================
# [SECTION 4] VEHICLE AGENT
# ==============================================================================
class VehicleAgent:
    """Represents a single vehicle agent executing the control logic."""

    def __init__(self, spec, pre_veh, dt, init_v, init_s, leader_spec):
        self.spec = spec
        self.id = spec['id']
        self.dt = dt
        self.h_min = spec.get('h_min', 0.5)

        # In S2, D represents purely internal delays for prediction
        self.D = spec['d_proc'] + spec['d_mech']
        self.tau = spec['tau']

        self.pre_veh_spec = pre_veh.spec if pre_veh else leader_spec
        self.pre_veh = pre_veh
        self.pre_tau = pre_veh.tau if pre_veh else 0.5
        self.pre_D = pre_veh.D if pre_veh else 0.0
        self.leader_tau = leader_spec['tau']

        self.params = {'alpha': spec['alpha'], 'b': spec['b'], 'c': spec['c']}

        # First-order Low Pass Filter (LPF) initialization
        self.tau_f = SimConfig.TAU_F
        self.lambda_filter = self.dt / (self.tau_f + self.dt)
        self.s_ref_filtered = init_s

        self.Gamma, self.B_i, self.B_pre = VehicleDynamics.get_system_matrices(self.tau, self.pre_tau)
        self.B_leader_ref = np.zeros(5); self.B_leader_ref[4] = 1.0 / self.leader_tau

        self.x = np.array([init_s, init_v, init_v, 0.0, 0.0])

        self.steps_D = int(round(self.D / dt))
        self.steps_pre_D = int(round(self.pre_D / dt))

        buffer_len = self.steps_D + 50
        self.u_buffer = deque([0.0]*buffer_len, maxlen=buffer_len)
        self.u_pre_buffer = deque([0.0]*buffer_len, maxlen=buffer_len)

        self.exp_kernels_full = [expm(self.Gamma * k * dt) for k in range(self.steps_D + 1)]

        self.is_recursive = False
        self.steps_diff = 0
        self.exp_kernels_gap = []

        # Extended Case Configuration (D_i > D_pre)
        if self.D > self.pre_D and self.pre_veh is not None:
            self.is_recursive = True
            self.steps_diff = self.steps_D - self.steps_pre_D
            self.exp_kernels_gap = [expm(self.Gamma * k * dt) for k in range(self.steps_diff + 1)]

        self.logs = {'t': [], 's': [], 'v': [], 'a': [], 'u': [], 's_ref': [], 'h_t': []}
        self.current_s_ref = init_s
        self.current_h_t = 0

        # Time-varying V2X communication delay initialization
        self.last_v2x_time = -1.0
        self.current_d_comm = spec['d_comm']

    def _integrate_convolution(self, kernels, B_matrix, input_buffer, steps_range):
        val = np.zeros(5)
        hist_list = list(input_buffer)
        for k in range(steps_range):
            idx = -(1 + k)
            if abs(idx) <= len(hist_list):
                val += np.dot(kernels[k], B_matrix) * hist_list[idx] * self.dt
        return val

    def compute_control(self, current_time, pre_u_curr):
        v_ego = self.x[1]

        # Time-varying V2X communication delay update (10Hz, Gaussian noise std=3ms)
        if current_time - self.last_v2x_time >= 0.1:
            comm_noise = np.random.normal(loc=0.0, scale=0.003)
            comm_noise = np.clip(comm_noise, -0.01, 0.01)
            self.current_d_comm = max(0.0, self.spec['d_comm'] + comm_noise)
            self.last_v2x_time = current_time

        current_spec = self.spec.copy()
        current_spec['d_comm'] = self.current_d_comm

        # 1. Raw S_ref and h_t computation
        s_ref_raw = SRefLogic.calculate_s_ref(
            v_ego, current_spec, self.pre_veh_spec, self.D
        )
        h_t_raw = SRefLogic.calculate_h_t(s_ref_raw, v_ego, self.h_min)

        # 2. Apply LPF to h_t to prevent abrupt gain scheduling
        if not hasattr(self, 'h_t_filtered'):
            self.h_t_filtered = h_t_raw

        self.h_t_filtered = self.lambda_filter * h_t_raw + (1 - self.lambda_filter) * self.h_t_filtered
        self.s_ref_filtered = self.h_t_filtered * max(v_ego, 1.0)

        # 3. Update control gain
        self.K = VehicleDynamics.get_control_gain(
            self.h_t_filtered, self.tau,
            self.params['alpha'], self.params['b'], self.params['c']
        )

        # 4. Predictor-Feedback Logic (Exact Predictor + Recursive compensation)
        self.u_pre_buffer.append(pre_u_curr)
        if self.pre_veh:
            self.x[2] = self.pre_veh.x[1]
            self.x[4] = self.pre_veh.x[3]

        q_zerostate = np.dot(self.exp_kernels_full[self.steps_D], self.x)

        q_input_own = self._integrate_convolution(
            self.exp_kernels_full, self.B_i, self.u_buffer, self.steps_D
        )

        q_input_neighbor = np.zeros(5)
        if not self.is_recursive:
            q_input_neighbor = self._integrate_convolution(
                self.exp_kernels_full, self.B_pre, self.u_pre_buffer, self.steps_D
            )
        else:
            q_neighbor_known = self._integrate_convolution(
                self.exp_kernels_full, self.B_pre, self.u_pre_buffer, self.steps_pre_D
            )
            phi_gap = self.exp_kernels_gap[self.steps_diff]
            q_shifted = np.dot(phi_gap, q_neighbor_known)

            q_gap_fill = np.zeros(5)
            for k in range(self.steps_diff):
                future_time = current_time + (self.D - k * self.dt)
                u_leader_ref = SimConfig.get_leader_accel(future_time)
                q_gap_fill += np.dot(self.exp_kernels_gap[k], self.B_leader_ref) * u_leader_ref * self.dt

            q_input_neighbor = q_shifted + q_gap_fill

        # 5. Compute final control input
        q_total = q_zerostate + q_input_own + q_input_neighbor
        u_cmd = np.dot(self.K, q_total)

        self.current_s_ref = self.s_ref_filtered
        self.current_h_t = self.h_t_filtered

        return u_cmd

    def update_physics(self, u_cmd):
        self.u_buffer.append(u_cmd)
        u_delayed = self.u_buffer[-(self.steps_D + 1)] if len(self.u_buffer) > self.steps_D else 0.0

        s, v, vp, a, ap = self.x
        ds = vp - v
        dv = a
        da = (-a + u_delayed) / self.tau

        self.x[0] += ds * self.dt
        self.x[1] += dv * self.dt
        self.x[3] += da * self.dt
        return self.x

    def save_log(self, t, u_cmd):
        self.logs['t'].append(t)
        self.logs['s'].append(self.x[0])
        self.logs['v'].append(self.x[1])
        self.logs['a'].append(self.x[3])
        self.logs['u'].append(u_cmd)
        self.logs['s_ref'].append(self.current_s_ref)
        self.logs['h_t'].append(self.current_h_t)


# ==============================================================================
# [SECTION 5] MAIN SIMULATION LOOP
# ==============================================================================
def run_simulation():
    print("Initializing Scenario 2 (Emergency Braking) Simulation...")

    t_warmup = np.arange(-SimConfig.WARMUP_TIME, 0, SimConfig.DT)
    t_main = np.arange(0, SimConfig.TOTAL_TIME, SimConfig.DT)
    t_total = np.concatenate((t_warmup, t_main))

    leader_v, leader_a = 25.0, 0.0
    vehicles = []
    pre_veh = None
    leader_spec = SimConfig.VEHICLE_SPECS[0]

    for spec in SimConfig.VEHICLE_SPECS:
        if spec['id'] == 0: continue
        temp_s_ref = SRefLogic.calculate_s_ref(leader_v, spec, leader_spec, spec['d_proc'] + spec['d_mech'])
        init_h_t = SRefLogic.calculate_h_t(temp_s_ref, leader_v, spec.get('h_min', 0.5))
        init_s = init_h_t * leader_v

        veh = VehicleAgent(spec, pre_veh, SimConfig.DT, leader_v, init_s, leader_spec)
        vehicles.append(veh)
        pre_veh = veh

    l_logs = {'t': [], 'v': [], 'a': []}

    for t in t_total:
        target_a = SimConfig.get_leader_accel(t)
        leader_a += ((target_a - leader_a) / 0.5) * SimConfig.DT
        leader_v += leader_a * SimConfig.DT

        current_pre_u = leader_a

        for i, veh in enumerate(vehicles):
            if i == 0:
                veh.x[2] = leader_v
                veh.x[4] = leader_a

            u_cmd = veh.compute_control(t, current_pre_u)
            veh.update_physics(u_cmd)

            if t >= 0:
                veh.save_log(t, u_cmd)

            current_pre_u = u_cmd

        if t >= 0:
            l_logs['t'].append(t)
            l_logs['v'].append(leader_v)
            l_logs['a'].append(leader_a)

    print("Simulation completed. Preparing analysis...\n")
    return l_logs, vehicles


# ==============================================================================
# [SECTION 6] PERFORMANCE ANALYZER
# ==============================================================================
def analyze_performance(l_logs, vehicles):
    print("="*85)
    print(f"{'Simulation Performance Analysis Report':^85}")
    print("="*85)
    print(f"{'Vehicle':<8} | {'Max Jerk':<10} | {'Max Accel':<10} | {'Min TTC':<9} | {'S-Viol':<9} | {'RMSE':<6} | {'Mean Spc':<10}")
    print(f"{'(ID)':<8} | {'(m/s^3)':<10} | {'(m/s^2)':<10} | {'(sec)':<9} | {'(%)':<9} | {'(m)':<6} | {'(m)':<10}")
    print("-" * 85)

    max_acc_leader = np.max(np.abs(l_logs['a']))
    mean_spaces = []

    for i, v in enumerate(vehicles):
        acc = np.array(v.logs['a'])
        jerk = np.diff(acc) / SimConfig.DT
        max_jerk = np.max(np.abs(jerk))
        max_acc = np.max(np.abs(acc))

        spacing = np.array(v.logs['s'])
        my_v = np.array(v.logs['v'])

        mean_space = np.mean(spacing)
        mean_spaces.append(mean_space)

        if i == 0:
            pre_v = np.array(l_logs['v'])
        else:
            pre_v = np.array(vehicles[i-1].logs['v'])

        rel_v = my_v - pre_v
        approaching_mask = rel_v > 0.1

        if np.any(approaching_mask):
            ttc_values = spacing[approaching_mask] / rel_v[approaching_mask]
            min_ttc = np.min(ttc_values)
        else:
            min_ttc = 99.9

        violation_count = np.sum(spacing < SimConfig.S0)
        total_steps = len(spacing)
        violation_rate = (violation_count / total_steps) * 100.0

        target_spacing = SimConfig.S0 + my_v * v.spec.get('h_min', 0.5)
        error = spacing - target_spacing
        rmse = np.sqrt(np.mean(error**2))

        ttc_str = f"{min_ttc:.2f}" if min_ttc < 99 else "Inf"
        print(f"Veh {v.id:<4} | {max_jerk:<10.2f} | {max_acc:<10.2f} | {ttc_str:<9} | {violation_rate:<8.1f}% | {rmse:<6.2f} | {mean_space:<10.2f}")

    print("-" * 85)

    avg_mean_space = np.mean(mean_spaces)
    print(f"Avg. Mean Space (All Vehicles): {avg_mean_space:.2f} m\n")

    last_veh_acc = np.max(np.abs(vehicles[-1].logs['a']))
    stability_ratio = last_veh_acc / max_acc_leader

    print(f"String Stability Ratio (Last/Leader): {stability_ratio:.4f}")
    if stability_ratio < 1.0:
        print(">> Result: STABLE (Shockwave attenuated)")
    else:
        print(">> Result: UNSTABLE (Shockwave amplified)")
    print("="*85 + "\n")


# ==============================================================================
# [SECTION 7] VISUALIZATION (PLOTS & ANIMATION)
# ==============================================================================
mpl.rcParams['animation.embed_limit'] = 100.0

def create_animation(t, l_logs, vehicles):
    print("Rendering animation video... (This may take a moment)")

    dt = t[1] - t[0]
    pos_leader = np.cumsum(l_logs['v']) * dt
    all_positions = [pos_leader]
    current_front_pos = pos_leader
    all_spacings = []

    for v in vehicles:
        spacing = np.array(v.logs['s'])
        min_len = min(len(current_front_pos), len(spacing))
        pos_i = current_front_pos[:min_len] - spacing[:min_len] - SimConfig.VEHICLE_LENGTH
        all_positions.append(pos_i)
        all_spacings.append(spacing)
        current_front_pos = pos_i

    skip_step = 20
    frame_indices = range(0, len(t), skip_step)

    fig, ax = plt.subplots(figsize=(14.0, 4))
    ax.set_facecolor('white')
    ax.set_ylim(-3.0, 7.0)
    ax.set_xlabel('Position (m)', color='black')
    ax.set_yticks([])
    ax.tick_params(axis='x', colors='black')
    ax.set_aspect('auto')

    lane_width = SimConfig.LANE_WIDTH
    ax.axhline(-lane_width/2, color='black', linewidth=2)
    ax.axhline(lane_width/2, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(lane_width * 1.5, color='black', linewidth=2)

    car_patches = []
    label_texts = []
    info_texts = []
    gap_texts = []
    colors = SimConfig.COLORS

    for i in range(len(all_positions)):
        c = colors[i % len(colors)]
        rect_y = -(SimConfig.VEHICLE_WIDTH / 2)
        rect = patches.Rectangle((0, rect_y), SimConfig.VEHICLE_LENGTH, SimConfig.VEHICLE_WIDTH,
                                 linewidth=1, edgecolor='black', facecolor=c)
        ax.add_patch(rect)
        car_patches.append(rect)

        label_txt = ax.text(0, 2.5, "", color=c, fontsize=16, fontweight='bold', ha='center')
        label_texts.append(label_txt)

        info_txt = ax.text(0, -2.8, "", color='black', fontsize=15, ha='center')
        info_texts.append(info_txt)

    for i in range(len(all_positions) - 1):
        gap_txt = ax.text(0, 0, "", color='darkgoldenrod', fontsize=15, fontweight='bold', ha='center', va='center')
        gap_texts.append(gap_txt)

    status_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, color='black', fontsize=20, fontweight='bold')

    def update(frame_idx):
        leader_x = all_positions[0][frame_idx]
        ax.set_xlim(leader_x - 130, leader_x + 15)

        curr_time = t[frame_idx]
        leader_vel = l_logs['v'][frame_idx]
        acc_str = "Accel" if l_logs['a'][frame_idx] > 0.1 else ("Decel" if l_logs['a'][frame_idx] < -0.1 else "Cruise")
        status_text.set_text(f"Time: {curr_time:.1f}s | Leader Vel: {leader_vel*3.6:.0f} km/h | State: {acc_str}")

        for i, pos_arr in enumerate(all_positions):
            if frame_idx < len(pos_arr):
                x_rear = pos_arr[frame_idx]
                car_patches[i].set_x(x_rear)

                label_texts[i].set_position((x_rear + SimConfig.VEHICLE_LENGTH/2, 2.2))
                label_texts[i].set_text("Leader" if i == 0 else f"Veh {i}")

                if i == 0: v_val = l_logs['v'][frame_idx]
                else: v_val = vehicles[i-1].logs['v'][frame_idx]

                info_texts[i].set_position((x_rear + SimConfig.VEHICLE_LENGTH/2, -2.5))
                info_texts[i].set_text(f"{v_val*3.6:.0f} km/h")

        for i in range(len(gap_texts)):
            if frame_idx < len(all_positions[i]) and frame_idx < len(all_positions[i+1]):
                front_rear = all_positions[i][frame_idx]
                rear_front = all_positions[i+1][frame_idx] + SimConfig.VEHICLE_LENGTH
                mid_x = (front_rear + rear_front) / 2
                gap_val = all_spacings[i][frame_idx]

                gap_texts[i].set_position((mid_x, 0))
                if gap_val < 3.0: gap_texts[i].set_text("")
                else: gap_texts[i].set_text(f"{gap_val:.1f}m")

        return car_patches + label_texts + info_texts + gap_texts + [status_text]

    ani = FuncAnimation(fig, update, frames=frame_indices, interval=50, blit=True)
    plt.close()
    return ani


if __name__ == "__main__":
    # 1. Execute Simulation
    l_logs, vehicles = run_simulation()

    # 2. Extract Performance Metrics
    analyze_performance(l_logs, vehicles)

    # 3. Static Plot Configuration
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'lines.linewidth': 1.5,
        'legend.framealpha': 0.6
    })

    t = l_logs['t']
    plt.figure(figsize=(6, 17))
    colors = SimConfig.COLORS

    # Plot 1: Acceleration
    plt.subplot(4, 1, 1)
    plt.plot(t, l_logs['a'], color=colors[0], linewidth=2.5, label='L', zorder=10)
    for i, v in enumerate(vehicles):
        plt.plot(t, v.logs['a'], color=colors[i+1], label=f'V{v.id}', alpha=0.8)
    plt.grid(True)
    plt.legend(loc='upper right').set_zorder(20)

    # Plot 2: Velocity
    plt.subplot(4, 1, 2)
    plt.plot(t, l_logs['v'], color=colors[0], linewidth=2.5, label='L', zorder=10)
    for i, v in enumerate(vehicles):
        plt.plot(t, v.logs['v'], color=colors[i+1], label=f'V{v.id}', alpha=0.8)
    plt.grid(True)
    plt.legend(loc='upper right').set_zorder(20)

    # Plot 3: Inter-Vehicle Spacing
    plt.subplot(4, 1, 3)
    for i, v in enumerate(vehicles):
        plt.plot(t, v.logs['s'], color=colors[i+1], linewidth=2.0, label=f'V{v.id}')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=18).set_zorder(20)

    # Plot 4: Effective Time Headway (h_t)
    plt.subplot(4, 1, 4)
    for i, v in enumerate(vehicles):
        plt.plot(t, v.logs['h_t'], color=colors[i+1], label=f'V{v.id}')
        # Individual h_min lower bounds
        plt.axhline(v.spec.get('h_min', 0.5), color=colors[i+1], linestyle='--', alpha=0.5, label=f'V{v.id}(H_MIN)')

    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.grid(True)

    # Reorder legend to show main curves first, then H_MIN bounds
    handles, labels = plt.gca().get_legend_handles_labels()
    main_handles, main_labels = [], []
    hmin_handles, hmin_labels = [], []

    for h, l in zip(handles, labels):
        if '(H_MIN)' in l:
            hmin_handles.append(h)
            hmin_labels.append(l)
        else:
            main_handles.append(h)
            main_labels.append(l)

    plt.legend(main_handles + hmin_handles, main_labels + hmin_labels, loc='upper right', fontsize=18).set_zorder(20)

    plt.tight_layout()
    plt.show()

    # 4. Generate Animation
    anim = create_animation(t, l_logs, vehicles)
    try:
        from IPython.display import display
        display(HTML(anim.to_jshtml()))
        print("Animation rendering complete!")
    except Exception as e:
        print(f"HTML rendering error: {e}")
