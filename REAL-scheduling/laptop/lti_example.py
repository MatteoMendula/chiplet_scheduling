import math

def evaluate_thermal_routing(node_constants, node_state, task_profile):
    """
    Evaluates if a task can be safely routed to a node based on LTI thermal modeling.
    """
    # 1. Unpack Node Constants
    T_amb = node_constants['T_amb']
    R_th = node_constants['R_th']
    theta = node_constants['theta']
    T_max = node_constants['T_max']
    
    # 2. Unpack Current Node State
    T_last = node_state['T_last']
    t_last = node_state['t_last']
    
    # 3. Unpack Task Profile (from MAESTRO)
    P = task_profile['P']
    T_exec = task_profile['T_exec']
    t_now = task_profile['t_now']
    
    # --- STEP 1: Calculate Cooling ---
    # How much time has passed since the node finished its last task?
    # (If the tasks are queued/overlapping, delta_t could be 0)
    delta_t = max(0.0, t_now - t_last)
    
    # LTI Cooling formula: Decay towards ambient temperature
    T_current = T_amb + (T_last - T_amb) * math.exp(-delta_t / theta)
    
    # --- STEP 2: Calculate Heating Spike ---
    # LTI Heating formula: B(1 - e^(-T_exec/theta)) where B = P * R_th
    B = P * R_th
    temp_spike = B * (1 - math.exp(-T_exec / theta))
    
    # --- STEP 3: Final Peak Prediction ---
    T_predicted_peak = T_current + temp_spike
    
    # --- STEP 4: Routing Decision ---
    is_safe = T_predicted_peak <= T_max
    
    # Calculate the new t_last (when this proposed task would finish)
    # If the node was idle, it starts at t_now. If it was busy, it queues up.
    new_t_last = max(t_now, t_last) + T_exec
    
    return {
        "is_safe": is_safe,
        "T_current_before_task": T_current,
        "T_predicted_peak": T_predicted_peak,
        "new_state_if_accepted": {
            "T_last": T_predicted_peak,
            "t_last": new_t_last
        }
    }

# ==========================================
# Example Usage:
# ==========================================

chiplet_node_constants = {
    "T_amb": 30.0,   # Hot summer day (30°C)
    "R_th": 12.0,    # 12 °C/W resistance
    "theta": 5.0,    # 5 seconds time constant
    "T_max": 70.0    # Throttle limit
}

# The chiplet ran a task a while ago and peaked at 65°C. 
# That task finished at second 100.
chiplet_node_state = {
    "T_last": 65.0,  
    "t_last": 100.0  
}

# A new frame arrives 3 seconds later (t=103)
# MAESTRO says Block 1 takes 3.5 Watts and runs for 0.8 seconds.
early_exit_block = {
    "P": 3.5,        
    "T_exec": 0.8,   
    "t_now": 103.0   
}

result = evaluate_thermal_routing(chiplet_node_constants, chiplet_node_state, early_exit_block)

print(f"Is safe to route? {result['is_safe']}")
print(f"Node cooled down to: {result['T_current_before_task']:.2f}°C")
print(f"Predicted peak temp: {result['T_predicted_peak']:.2f}°C")