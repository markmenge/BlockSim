# filename: sim_blocks.py
# Additions for Phase 5: Simulation Engine

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from collections import defaultdict, deque # Add deque for Kahn's algorithm

# --- Base Class (SimBlock - unchanged from Phase 1) ---
class SimBlock(ABC):
    # (Keep existing SimBlock definition with get_ports, get_parameters, update, on_simulation_end)
    # ...
    @abstractmethod
    def get_ports(self):
        raise NotImplementedError()
    def get_parameters(self):
        return {}
    @abstractmethod
    def update(self, t, dt, inputs):
        raise NotImplementedError()
    def on_simulation_end(self):
        pass

# --- Block Implementations (Constant, Sum, Integrator, Plot - unchanged) ---
class Constant(SimBlock):
    # ... (no changes) ...
    def __init__(self, value=1.0): super().__init__(); self.value = float(value)
    def get_ports(self): return {'inputs': {}, 'outputs': {'out': 'float'}}
    def get_parameters(self): return {'value': 'float'}
    def update(self, t, dt, inputs): return {'out': self.value}

class Sum(SimBlock):
    # ... (no changes) ...
    def __init__(self): super().__init__()
    def get_ports(self): return {'inputs': {'in1': 'float', 'in2': 'float'}, 'outputs': {'out': 'float'}}
    def update(self, t, dt, inputs):
        in1 = float(inputs.get('in1', 0.0)); in2 = float(inputs.get('in2', 0.0))
        return {'out': in1 + in2}

class Integrator(SimBlock):
    # ... (no changes) ...
    def __init__(self, initial=0.0):
        super().__init__(); self.initial = float(initial); self.state = float(initial)
    def get_ports(self): return {'inputs': {'signal': 'float'}, 'outputs': {'state': 'float'}}
    def get_parameters(self): return {'initial': 'float'}
    def update(self, t, dt, inputs):
        input_signal = float(inputs.get('signal', 0.0)); self.state += input_signal * dt
        return {'state': self.state}

class Plot(SimBlock):
    # ... (no changes needed, uses on_simulation_end) ...
    def __init__(self, title="Plot Output", ylabel="Value", num_inputs=1):
        super().__init__(); self.plot_title = title; self.plot_ylabel = ylabel
        self.num_inputs = max(1, int(num_inputs)); self.t_data = []
        self.y_data = [[] for _ in range(self.num_inputs)]
    def get_ports(self):
        return {'inputs': {f'in{i+1}': 'any' for i in range(self.num_inputs)}, 'outputs': {}}
    def get_parameters(self): return {'title': 'string', 'ylabel': 'string', 'num_inputs': 'int'}
    def update(self, t, dt, inputs):
        self.t_data.append(t)
        for i in range(self.num_inputs):
            self.y_data[i].append(inputs.get(f'in{i+1}', None))
        return {}
    def on_simulation_end(self):
        print(f"Plotting data for '{self.plot_title}'...")
        # ... (rest of plotting logic) ...
        if not self.t_data or all(not y for y in self.y_data):
            print("  No data collected for plotting."); return
        plt.figure()
        has_data = False
        for i in range(self.num_inputs):
            valid_t = [t for t, y in zip(self.t_data, self.y_data[i]) if y is not None]
            valid_y = [y for y in self.y_data[i] if y is not None]
            if valid_t:
                label = f"Input {i+1}" if self.num_inputs > 1 else "Input Signal"
                plt.plot(valid_t, valid_y, label=label)
                has_data = True
        if not has_data: print("  No valid data points found across all inputs."); plt.close(); return
        plt.xlabel("Time (s)"); plt.ylabel(self.plot_ylabel); plt.title(self.plot_title)
        plt.grid(True)
        if self.num_inputs > 1: plt.legend()
        try: plt.show()
        except Exception as e: print(f"Error displaying plot: {e}")


# --- Phase 5: Simulation Engine ---

class SimulationEngine:
    """
    Manages the execution of a BlockSim model based on block instances
    and explicit port connections.
    """
    def __init__(self, block_instances_map, connections_list):
        """
        Initializes the simulation engine.

        Args:
            block_instances_map (dict): Map of {block_id: SimBlock_instance}.
            connections_list (list): List of connections, where each item is
                                     [src_id, src_port, dst_id, dst_port].
        """
        self.blocks = block_instances_map # {block_id: instance}
        self.raw_connections = connections_list
        self.connections = defaultdict(dict) # Processed: {dst_id: {dst_port: (src_id, src_port)}}
        self.execution_order = [] # List of block_ids

        print("Initializing Simulation Engine...")
        self._build_graph()
        self._topological_sort()
        print(f"Execution Order: {self.execution_order}")

    def _build_graph(self):
        """Processes raw connections into an efficient lookup structure."""
        print("Building connection graph...")
        for conn in self.raw_connections:
            src_id, src_port, dst_id, dst_port = conn

            # Validation
            if src_id not in self.blocks:
                print(f"Warning: Source block ID {src_id} in connection not found in instances. Skipping connection.")
                continue
            if dst_id not in self.blocks:
                print(f"Warning: Destination block ID {dst_id} in connection not found in instances. Skipping connection.")
                continue

            src_block = self.blocks[src_id]
            dst_block = self.blocks[dst_id]

            # Check if ports exist
            src_ports = src_block.get_ports().get('outputs', {})
            dst_ports = dst_block.get_ports().get('inputs', {})

            if src_port not in src_ports:
                print(f"Warning: Source port '{src_port}' not found in outputs of block {src_id} ({type(src_block).__name__}). Skipping connection.")
                continue
            if dst_port not in dst_ports:
                print(f"Warning: Destination port '{dst_port}' not found in inputs of block {dst_id} ({type(dst_block).__name__}). Skipping connection.")
                continue

            # Check for multiple connections to the same input port
            if dst_port in self.connections[dst_id]:
                 existing_src_id, existing_src_port = self.connections[dst_id][dst_port]
                 print(f"Warning: Input port '{dst_port}' on block {dst_id} already connected to {existing_src_id}:{existing_src_port}. Overwriting with connection from {src_id}:{src_port}.")

            # Store connection: Destination block -> input port -> (Source block, output port)
            self.connections[dst_id][dst_port] = (src_id, src_port)
        print("Connection graph built.")

    def _topological_sort(self):
        """
        Calculates the block execution order using Kahn's algorithm.
        Detects cycles.
        """
        print("Calculating execution order (Topological Sort)...")
        in_degree = {block_id: 0 for block_id in self.blocks}
        # Adjacency list: source_block -> list of destination_blocks
        adj = defaultdict(list)

        # Calculate in-degrees and build adjacency list
        for src_id in self.blocks:
            # Find all blocks that *receive* input from src_id
            for dst_id, input_ports in self.connections.items():
                 for dst_port, (conn_src_id, _) in input_ports.items():
                      if conn_src_id == src_id:
                           if dst_id not in adj[src_id]: # Avoid duplicates in adj list
                                adj[src_id].append(dst_id)
                           in_degree[dst_id] += 1 # Increment in-degree of the destination

        # Initialize queue with nodes having in-degree 0
        queue = deque([block_id for block_id in self.blocks if in_degree[block_id] == 0])
        self.execution_order = []
        count = 0 # Count of visited nodes

        while queue:
            u = queue.popleft()
            self.execution_order.append(u)
            count += 1

            # For each neighbor v of u, decrease its in-degree
            for v in adj[u]:
                in_degree[v] -= 1
                # If in-degree becomes 0, add it to the queue
                if in_degree[v] == 0:
                    queue.append(v)

        # Check for cycles
        if count != len(self.blocks):
            # Find nodes involved in the cycle (those with in_degree > 0) for better error message
            cycle_nodes = [bid for bid, degree in in_degree.items() if degree > 0]
            error_msg = f"Cycle detected in the block graph. Cannot determine execution order. Nodes with remaining in-degree: {cycle_nodes}"
            print(f"ERROR: {error_msg}")
            # Clear execution order to prevent running
            self.execution_order = []
            raise ValueError(error_msg)
        else:
            print("Topological sort successful.")


    def run(self, t_stop, dt):
        """
        Runs the simulation loop.

        Args:
            t_stop (float): Simulation end time.
            dt (float): Simulation time step.
        """
        if not self.execution_order:
             print("Error: Cannot run simulation. Invalid execution order (possibly due to cycles or build errors).")
             return

        print(f"--- Running Simulation Engine (t_stop={t_stop}, dt={dt}) ---")
        t = 0.0
        # Stores the outputs of all blocks for the *current* time step
        # Structure: {block_id: {port_name: value, ...}}
        current_outputs = {block_id: {} for block_id in self.blocks}

        # --- Simulation Time Loop ---
        while t <= t_stop:
            # --- Evaluate blocks in topological order ---
            for block_id in self.execution_order:
                block = self.blocks[block_id]
                inputs_for_block = {}

                # Gather inputs for this block based on connections
                if block_id in self.connections:
                    for dst_port, (src_id, src_port) in self.connections[block_id].items():
                        # Get the output from the source block computed *earlier* in this step
                        if src_id in current_outputs and src_port in current_outputs[src_id]:
                            inputs_for_block[dst_port] = current_outputs[src_id][src_port]
                        else:
                            # Source block or port hasn't produced output (shouldn't happen with topo sort)
                            # Or input is simply unconnected - use default (None or 0?)
                            # Current SimBlock update methods handle missing keys with .get()
                            # print(f"Debug: Input {block_id}:{dst_port} from {src_id}:{src_port} - source output not found yet.")
                            pass # Let the block's update handle missing input via .get()

                # Execute the block's update logic
                try:
                    outputs = block.update(t, dt, inputs_for_block)
                    # Store the outputs produced by this block in this time step
                    current_outputs[block_id] = outputs if outputs else {} # Ensure it's always a dict
                except Exception as e:
                    print(f"ERROR during update of block {block_id} ({type(block).__name__}) at t={t:.3f}: {e}")
                    # Option: Stop simulation or try to continue? Stop is safer.
                    print("Simulation stopped due to error.")
                    # Optionally call end hooks even on error?
                    self._call_end_hooks()
                    raise # Re-raise the exception

            # --- Time Increment ---
            t += dt

        print(f"--- Simulation Loop Finished at t={t-dt:.3f} ---")

        # --- Call end-of-simulation hooks ---
        self._call_end_hooks()

        print("--- Simulation Engine Finished ---")

    def _call_end_hooks(self):
        """Calls the on_simulation_end method for all blocks."""
        print("--- Calling on_simulation_end hooks ---")
        for block_id in self.blocks: # Call in arbitrary order, usually fine
             try:
                 self.blocks[block_id].on_simulation_end()
             except Exception as e:
                 print(f"Error during on_simulation_end for block {block_id} ({type(self.blocks[block_id]).__name__}): {e}")


# --- Global simulate function (Now uses the Engine) ---

def simulate(block_instances_map, connections_list, t_stop, dt):
    """
    High-level function to run a simulation using the SimulationEngine.

    Args:
        block_instances_map (dict): Map of {block_id: SimBlock_instance}.
        connections_list (list): List of connections [src_id, src_port, dst_id, dst_port].
        t_stop (float): Simulation end time.
        dt (float): Simulation time step.
    """
    try:
        engine = SimulationEngine(block_instances_map, connections_list)
        engine.run(t_stop, dt)
    except ValueError as e: # Catch cycle errors etc. from engine init
        print(f"Simulation setup failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()

# --- Example Usage (If run directly - Updated) ---
if __name__ == "__main__":
    print("Running sim_blocks.py engine example...")

    # Instantiate blocks (Assign arbitrary IDs for the map)
    # In GUI, these IDs come from the loaded model
    block_map = {
        1: Constant(value=1.0),
        2: Integrator(initial=0.0),
        3: Plot(title="Engine Test", ylabel="Integrated Value")
    }

    # Define connections using block IDs and port names
    # Connection: 1['out'] -> 2['signal']
    # Connection: 2['state'] -> 3['in1']
    conn_list = [
        [1, 'out', 2, 'signal'],
        [2, 'state', 3, 'in1']
    ]

    # Simulate using the global function which now uses the engine
    simulate(block_map, conn_list, t_stop=5.0, dt=0.01)

    print("\n--- Example Complete ---")
    