from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
import serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# we need to import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
    return vehicle_per_lane


def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)



class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TrafficLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # output for all time steps
        out = self.fc(out[:, -1, :])     # use output of the last time step
        return out
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        # ---- Channel Attention ----
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # ---- Spatial Attention ----
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                                      stride=1, padding=kernel_size // 2, bias=False)
        
    def forward(self, x):
        # Channel attention
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out, _ = torch.max(x, dim=(2, 3), keepdim=True)
        avg_out = self.mlp(avg_out.view(x.size(0), -1))
        max_out = self.mlp(max_out.view(x.size(0), -1))
        channel_attention = torch.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        x = x * channel_attention

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attention

        return x
class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        junctions,
        max_memory_size=100000,
        epsilon_dec=5e-4,
        epsilon_end=0.05,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = Model(
            self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions
        )
        self.memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "new_state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "reward_memory":np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=bool),
                "mem_cntr": 0,
                "iter_cntr": 0,
            }


    def store_transition(self, state, state_, action,reward, done,junction):
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def reset(self,junction_numbers):
        for junction_number in junction_numbers:
            self.memory[junction_number]['mem_cntr'] = 0

    def save(self,model_name):
        torch.save(self.Q_eval.state_dict(),f'models/{model_name}.bin')

    def learn(self, junction):
        self.Q_eval.optimizer.zero_grad()

        batch= np.arange(self.memory[junction]['mem_cntr'], dtype=np.int32)

        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(
            self.Q_eval.device
        )
        new_state_batch = torch.tensor(
            self.memory[junction]["new_state_memory"][batch]
        ).to(self.Q_eval.device)
        reward_batch = torch.tensor(
            self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]["action_memory"][batch]

        q_eval = self.Q_eval.forward(state_batch)[batch, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = (
            self.epsilon - self.epsilon_dec
            if self.epsilon > self.epsilon_end
            else self.epsilon_end
        )


def run(train=True, model_name="model", epochs=50, steps=500, ard=False):
    if ard:
        arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
        def write_read(x):
            arduino.write(bytes(x, 'utf-8'))
            time.sleep(0.05)
            data = arduino.readline()
            return data

    best_time = np.inf
    total_time_list = []

    # Get junctions
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"])
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    traci.close()

    # Initialize RL Agent
    brain = Agent(
        gamma=0.99,
        epsilon=0.0,
        lr=0.1,
        input_dims=4,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions=4,
        junctions=junction_numbers,
    )

    if not train:
        brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin', map_location=brain.Q_eval.device))

    for e in range(epochs):
        sumo_binary = checkBinary("sumo") if train else checkBinary("sumo-gui")
        traci.start([sumo_binary, "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        print(f"Epoch {e+1}/{epochs}")

        select_lane = [
            ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
            ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
            ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
            ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
        ]

        step = 0
        total_time = 0
        min_duration = 5
        traffic_lights_time = {j: 0 for j in all_junctions}
        prev_action = {j_num: 0 for j_num in junction_numbers}
        prev_vehicles_per_lane = {j_num: [0]*4 for j_num in junction_numbers}
        all_lanes = [l for j in all_junctions for l in traci.trafficlight.getControlledLanes(j)]

        # Vehicle tracking
        veh_start_time, veh_end_time = {}, {}
        veh_wait_time, veh_total_wait_time = {}, {}
        queue_lengths = []

        # Junction tracking
        junction_vehicle_wait = {j: [] for j in all_junctions}
        junction_queue_history = {j: [] for j in all_junctions}

        while step <= steps:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            # Track departures
            for vid in traci.simulation.getDepartedIDList():
                veh_start_time[vid] = sim_time
                veh_wait_time[vid] = 0.0
                veh_total_wait_time[vid] = 0.0

            # Track arrivals
            for vid in traci.simulation.getArrivedIDList():
                veh_end_time[vid] = sim_time

            # Update waiting time per vehicle
            for vid in traci.vehicle.getIDList():
                prev_wait = veh_wait_time.get(vid, 0.0)
                current_wait = traci.vehicle.getWaitingTime(vid)
                delta_wait = current_wait - prev_wait
                veh_wait_time[vid] = current_wait
                veh_total_wait_time[vid] = veh_total_wait_time.get(vid, 0.0) + delta_wait

                # Assign delta wait to junctions
                lane_id = traci.vehicle.getLaneID(vid)
                for junction in all_junctions:
                    if lane_id in traci.trafficlight.getControlledLanes(junction):
                        junction_vehicle_wait[junction].append(delta_wait)
                        break

            # Track overall queue length
            total_halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)
            queue_lengths.append(total_halting)

            # Junction RL control
            for junction_number, junction in enumerate(all_junctions):
                lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(lanes)
                total_time += waiting_time

                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(lanes)
                    reward = -1 * waiting_time
                    state_ = list(vehicles_per_lane.values())
                    state = prev_vehicles_per_lane[junction_number]
                    prev_vehicles_per_lane[junction_number] = state_

                    brain.store_transition(state, state_, prev_action[junction_number], reward, (step==steps), junction_number)
                    lane = brain.choose_action(state_)
                    prev_action[junction_number] = lane

                    phaseDuration(junction, 6, select_lane[lane][0])
                    phaseDuration(junction, min_duration + 10, select_lane[lane][1])



                    if ard:
                        ph = str(traci.trafficlight.getPhase("0"))
                        value = write_read(ph)

                    traffic_lights_time[junction] = min_duration + 10
                    if train:
                        brain.learn(junction_number)
                else:
                    traffic_lights_time[junction] -= 1

                # Per-junction queue
                avg_queue = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
                junction_queue_history[junction].append(avg_queue)

            step += 1

        # --- Metrics calculation ---
        total_wait = sum(veh_total_wait_time.values())
        total_turnaround = sum(veh_end_time[vid]-veh_start_time[vid] for vid in veh_end_time if vid in veh_start_time)
        count = len(veh_total_wait_time)

        avg_waiting_time = total_wait / count if count else 0
        avg_turnaround_time = total_turnaround / count if count else 0
        avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0

        # --- Print per-junction metrics ---
        print("---- PER-JUNCTION PERFORMANCE ----")
        for junction in all_junctions:
            awt_junction = np.mean(junction_vehicle_wait[junction]) if junction_vehicle_wait[junction] else 0
            aql_junction = np.mean(junction_queue_history[junction]) if junction_queue_history[junction] else 0
            print(f"Junction {junction}: AWT={awt_junction:.2f}s | AQL={aql_junction:.2f}")

        # --- Print overall metrics ---
        print("---- OVERALL PERFORMANCE ----")
        print(f"Average Waiting Time (AWT): {avg_waiting_time:.2f} s")
        print(f"Average Turnaround Time (ATT): {avg_turnaround_time:.2f} s")
        print(f"Average Queue Length (AQL): {avg_queue_length:.2f} vehicles")
        print(f"Total waiting time (all vehicles): {total_time:.2f}")
        print("-----------------------------------\n")
        

        total_time_list.append(total_time)
        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save(model_name)

        traci.close()
        sys.stdout.flush()
        if not train:
            break

    if train:
        plt.plot(list(range(len(total_time_list))), total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total waiting time")
        plt.title("Total Waiting Time vs Epoch")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()



def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action = 'store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    optParser.add_option(
       "--ard",
        action='store_true',
        default=False,
        help="Connect Arduino", 
    )
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    ard = options.ard
    run(train=train,model_name=model_name,epochs=epochs,steps=steps,ard=ard)
