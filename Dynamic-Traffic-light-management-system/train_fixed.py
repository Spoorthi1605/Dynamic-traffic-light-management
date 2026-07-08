from __future__ import absolute_import, print_function

import os
import sys
import time
import optparse
import numpy as np
import pandas as pd
from collections import defaultdict

# SUMO tools
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci


def generate_standard_phases(junction, num_lanes):
    """Generate standard 4-phase traffic light pattern for a junction"""
    
    if num_lanes == 12:  # Typical 4-approach, 3-lane intersection
        phase1 = "GGGrrrGGGrrr"  # NS straight + right
        phase2 = "rrrGGGrrrrrr"  # NS left (protected)
        phase3 = "rrrrrrGGGrrr"  # EW straight + right  
        phase4 = "rrrrrrrrrGGG"  # EW left (protected)
    elif num_lanes == 8:  # Smaller intersection
        phase1 = "GGrrGGrr"      # NS straight + right
        phase2 = "rrGGrrrr"      # NS left
        phase3 = "rrrrGGrr"      # EW straight + right  
        phase4 = "rrrrrrGG"      # EW left
    else:  # Default pattern
        base_phase = ['r'] * num_lanes
        phases = []
        for i in range(4):
            phase = base_phase.copy()
            start = i * num_lanes // 4
            end = (i + 1) * num_lanes // 4
            for j in range(start, end):
                phase[j] = 'G'
            phases.append(''.join(phase))
        return phases
    
    return [phase1, phase2, phase3, phase4]


def run_fixed_time_with_simple_metrics(steps=1000, phase_duration=30):
    """Run fixed-time traffic signal control with simple AWT, TTAT, AQL metrics"""
    sumo_binary = checkBinary("sumo-gui")
    traci.start([sumo_binary, "-c", "configuration.sumocfg", 
                 "--tripinfo-output", "tripinfo.xml",
                 "--waiting-time-memory", "1000"])
    
    all_junctions = traci.trafficlight.getIDList()
    
    # Initialize phase configurations
    junction_configs = {}
    for junction in all_junctions:
        lanes = traci.trafficlight.getControlledLanes(junction)
        num_lanes = len(lanes)
        phases = generate_standard_phases(junction, num_lanes)
        
        junction_configs[junction] = {
            'phases': phases,
            'current_phase': 0,
            'remaining_time': phase_duration
        }
    
    # === SIMPLE METRICS TRACKING (YOUR APPROACH) ===
    step = 0
    total_time = 0
    all_lanes = [l for j in all_junctions for l in traci.trafficlight.getControlledLanes(j)]

    # Vehicle tracking
    veh_start_time, veh_end_time = {}, {}
    veh_wait_time, veh_total_wait_time = {}, {}
    queue_lengths = []

    # Junction tracking
    junction_vehicle_wait = {j: [] for j in all_junctions}
    junction_queue_history = {j: [] for j in all_junctions}

    print(f"Starting simulation with {len(all_junctions)} junctions")
    print(f"Phase duration: {phase_duration} steps")
    print(f"Total simulation steps: {steps}")
    
    while step <= steps:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # === YOUR ORIGINAL METRICS TRACKING CODE ===
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

        # Update traffic lights
        for junction in all_junctions:
            config = junction_configs[junction]
            config['remaining_time'] -= 1
            
            if config['remaining_time'] <= 0:
                config['current_phase'] = (config['current_phase'] + 1) % len(config['phases'])
                config['remaining_time'] = phase_duration
            
            current_phase = config['phases'][config['current_phase']]
            traci.trafficlight.setRedYellowGreenState(junction, current_phase)

            # Per-junction queue tracking
            lanes = traci.trafficlight.getControlledLanes(junction)
            avg_queue = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
            junction_queue_history[junction].append(avg_queue)
        
        step += 1

    # === YOUR ORIGINAL METRICS CALCULATION ===
    total_wait = sum(veh_total_wait_time.values())
    total_turnaround = sum(veh_end_time[vid]-veh_start_time[vid] for vid in veh_end_time if vid in veh_start_time)
    count = len(veh_total_wait_time)

    avg_waiting_time = total_wait / count if count else 0
    avg_turnaround_time = total_turnaround / count if count else 0
    avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0

    # === PRINT RESULTS IN YOUR FORMAT ===
    print("\n---- PER-JUNCTION PERFORMANCE ----")
    for junction in all_junctions:
        awt_junction = np.mean(junction_vehicle_wait[junction]) if junction_vehicle_wait[junction] else 0
        aql_junction = np.mean(junction_queue_history[junction]) if junction_queue_history[junction] else 0
        print(f"Junction {junction}: AWT={awt_junction:.2f}s | AQL={aql_junction:.2f}")

    print("\n---- OVERALL PERFORMANCE ----")
    print(f"Average Waiting Time (AWT): {avg_waiting_time:.2f} s")
    print(f"Average Turnaround Time (ATT): {avg_turnaround_time:.2f} s")
    print(f"Average Queue Length (AQL): {avg_queue_length:.2f} vehicles")
    print(f"Total waiting time (all vehicles): {total_time:.2f}")
    print("-----------------------------------\n")

    # Save simple metrics to CSV
    save_simple_metrics(veh_start_time, veh_end_time, veh_total_wait_time, all_junctions)

    traci.close()


def save_simple_metrics(veh_start_time, veh_end_time, veh_total_wait_time, all_junctions):
    """Save simple metrics in your format"""
    metrics_data = []
    
    for vid, wait_time in veh_total_wait_time.items():
        start_time = veh_start_time.get(vid, 0)
        end_time = veh_end_time.get(vid, 0)
        travel_time = end_time - start_time if end_time > 0 else 0
        
        metrics_data.append({
            'vehicle_id': vid,
            'waiting_time': wait_time,
            'travel_time': travel_time,
            'completed': vid in veh_end_time
        })
    
    df = pd.DataFrame(metrics_data)
    df.to_csv('simple_traffic_metrics.csv', index=False)
    print("Simple metrics saved to 'simple_traffic_metrics.csv'")


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-s", dest='steps', type='int', default=1000, help="Number of steps")
    optParser.add_option("-d", dest='duration', type='int', default=30, help="Phase duration")
    options, args = optParser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    run_fixed_time_with_simple_metrics(steps=options.steps, phase_duration=options.duration)