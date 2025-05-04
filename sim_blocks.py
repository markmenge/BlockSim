# filename: sim_blocks.py
# Required packages:
#   pip install matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# --- Base Class ---

class SimBlock(ABC):
    """
    Abstract base class for all simulation blocks in BlockSim.
    Defines the required interface for metadata and simulation updates.
    """

    @abstractmethod
    def get_ports(self):
        """
        Returns a dictionary describing the block's input and output ports.

        Returns:
            dict: {'inputs': {'port_name': type_str, ...},
                   'outputs': {'port_name': type_str, ...}}
                  Type strings are informational for now (e.g., 'float', 'int', 'any').
        """
        raise NotImplementedError()

    def get_parameters(self):
        """
        Returns a dictionary describing the block's configurable parameters.
        These are typically set at instantiation time.

        Returns:
            dict: {'param_name': type_str, ...}
                  Returns an empty dict if the block has no parameters.
        """
        return {} # Default: no parameters

    @abstractmethod
    def update(self, t, dt, inputs):
        """
        Executes the block's logic for a single time step.

        Args:
            t (float): Current simulation time.
            dt (float): Simulation time step.
            inputs (dict): A dictionary where keys are the block's defined
                           input port names and values are the signals
                           connected to those ports for this timestep.
                           {'port_name': value, ...}

        Returns:
            dict: A dictionary where keys are the block's defined
                  output port names and values are the calculated outputs
                  for this timestep. {'port_name': value, ...}
                  Returns an empty dict if the block has no outputs.
        """
        raise NotImplementedError()

    def on_simulation_end(self):
        """
        Optional hook called once after the simulation loop finishes.
        Useful for blocks that need to perform final actions, like plotting.
        Does nothing by default.
        """
        pass # Default behavior is to do nothing

# --- Block Implementations ---

class Constant(SimBlock):
    """
    Outputs a constant value.
    """
    def __init__(self, value=1.0):
        super().__init__()
        self.value = float(value)

    def get_ports(self):
        """Defines one output port."""
        return {
            'inputs': {},
            'outputs': {'out': 'float'}
        }

    def get_parameters(self):
        """Defines the 'value' parameter."""
        return {'value': 'float'}

    def update(self, t, dt, inputs):
        """Returns the constant value on the 'out' port."""
        # inputs dict is ignored as this block has no inputs
        return {'out': self.value}

class Sum(SimBlock):
    """
    Adds two input signals together. Defaults inputs to 0 if not connected.
    """
    def __init__(self):
        super().__init__()
        # No parameters needed for basic 2-input sum

    def get_ports(self):
        """Defines two input ports ('in1', 'in2') and one output port ('out')."""
        return {
            'inputs': {'in1': 'float', 'in2': 'float'},
            'outputs': {'out': 'float'}
        }

    # No get_parameters needed (returns {} by default)

    def update(self, t, dt, inputs):
        """Calculates sum = in1 + in2. Uses 0 for unconnected inputs."""
        in1_value = inputs.get('in1', 0.0) # Default to 0 if 'in1' key is missing
        in2_value = inputs.get('in2', 0.0) # Default to 0 if 'in2' key is missing
        sum_value = float(in1_value) + float(in2_value)
        return {'out': sum_value}

class Integrator(SimBlock):
    """
    Computes the integral of an input signal over time using Euler method.
    """
    def __init__(self, initial=0.0):
        super().__init__()
        self.initial = float(initial)
        self.state = float(initial)

    def get_ports(self):
        """Defines one input ('signal') and one output ('state')."""
        return {
            'inputs': {'signal': 'float'},
            'outputs': {'state': 'float'}
        }

    def get_parameters(self):
        """Defines the 'initial' value parameter."""
        return {'initial': 'float'}

    def update(self, t, dt, inputs):
        """Updates the internal state based on the input signal."""
        input_signal = inputs.get('signal', 0.0) # Default to 0 if unconnected
        self.state += float(input_signal) * dt
        return {'state': self.state}

class Plot(SimBlock):
    """
    Collects data from its input port during simulation and generates a plot
    at the end of the simulation.
    """
    def __init__(self, title="Plot Output", ylabel="Value", num_inputs=1):
        super().__init__()
        self.plot_title = title
        self.plot_ylabel = ylabel
        self.num_inputs = max(1, int(num_inputs)) # Ensure at least one input

        # Data storage
        self.t_data = []
        # Store data for multiple inputs if needed
        self.y_data = [[] for _ in range(self.num_inputs)]

    def get_ports(self):
        """Defines input port(s) 'in1', 'in2', ... and no outputs."""
        input_ports = {f'in{i+1}': 'any' for i in range(self.num_inputs)}
        return {
            'inputs': input_ports,
            'outputs': {}
        }

    def get_parameters(self):
        """Defines plot configuration parameters."""
        return {'title': 'string', 'ylabel': 'string', 'num_inputs': 'int'}

    def update(self, t, dt, inputs):
        """Appends the current time and input value(s) to internal lists."""
        self.t_data.append(t)
        for i in range(self.num_inputs):
            port_name = f'in{i+1}'
            # Append None if input is not connected or value is missing
            value = inputs.get(port_name, None)
            self.y_data[i].append(value)
        return {} # No outputs

    def on_simulation_end(self):
        """Generates and displays the plot using matplotlib."""
        print(f"Plotting data for '{self.plot_title}'...")
        if not self.t_data or all(not y for y in self.y_data):
            print("  No data collected for plotting.")
            return

        plt.figure()
        for i in range(self.num_inputs):
            # Only plot if there's actual data for this input
            # Filter out None values which indicate missing data points
            valid_t = [t for t, y in zip(self.t_data, self.y_data[i]) if y is not None]
            valid_y = [y for y in self.y_data[i] if y is not None]

            if valid_t:
                label = f"Input {i+1}" if self.num_inputs > 1 else "Input Signal"
                plt.plot(valid_t, valid_y, label=label)
            else:
                print(f"  No valid data points for input {i+1}.")


        plt.xlabel("Time (s)")
        plt.ylabel(self.plot_ylabel)
        plt.title(self.plot_title)
        plt.grid(True)
        if self.num_inputs > 1 and any(valid_t for valid_t, _ in zip(self.t_data, self.y_data[0])): # Only show legend if useful
            plt.legend()
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")
            print("Ensure you have a suitable matplotlib backend configured.")


# --- Simulation Loop (Placeholder - Needs Graph Engine) ---

def simulate(block_instances, t_stop, dt):
    """
    (Placeholder) Basic discrete-time simulation loop.

    *** NOTE: This function needs to be replaced by a graph-based simulation
    *** engine that understands block connections via named ports and executes
    *** blocks in the correct topological order. This current version does NOT
    *** correctly handle data flow based on the new port structure.

    Args:
        block_instances (list): A list containing instances of SimBlock subclasses.
                                (Order may not matter once topological sort is used).
        t_stop (float): Simulation end time.
        dt (float): Simulation time step.
    """
    print("--- Starting Simulation (Placeholder Engine) ---")
    t = 0.0

    # --- Pre-computation / Graph Building (Required for real engine) ---
    # 1. Build connection graph: which output port connects to which input port?
    # 2. Perform topological sort to get execution order.
    # 3. Initialize block states and outputs.
    print("Warning: Using placeholder simulation loop. Data flow via ports is NOT implemented.")
    print("Block execution order in this loop is arbitrary.")

    # --- Simulation Time Loop ---
    while t <= t_stop:
        # --- In a real engine: ---
        # 1. Iterate through blocks in *topological order*.
        # 2. For each block, gather inputs from the outputs computed *earlier* in this step.
        # 3. Call block.update(t, dt, gathered_inputs_dict).
        # 4. Store the block's outputs for downstream blocks in *this* step.

        # --- Placeholder execution (calls update on all blocks arbitrarily): ---
        all_inputs_this_step = {} # In real engine, this would come from connections
        all_outputs_this_step = {}
        for block in block_instances:
             # This is incorrect - inputs aren't passed correctly
             outputs = block.update(t, dt, all_inputs_this_step)
             # In real engine, store outputs keyed by block ID and port name
             # For placeholder, just collect them (not useful)
             all_outputs_this_step[block] = outputs # Example structure

        # --- Time Increment ---
        t += dt

    print(f"--- Simulation Loop Finished at t={t-dt:.3f} ---")

    # --- Call end-of-simulation hooks ---
    print("--- Calling on_simulation_end hooks ---")
    for block in block_instances:
        try:
            block.on_simulation_end()
        except Exception as e:
            print(f"Error during on_simulation_end for {type(block).__name__}: {e}")

    print("--- Simulation Complete ---")

# --- Example Usage (If run directly) ---
if __name__ == "__main__":
    print("Running simple sim_blocks.py example...")

    # Instantiate blocks
    c1 = Constant(value=1.0)
    integ1 = Integrator(initial=0.0)
    plot1 = Plot(title="Integrator Test", ylabel="Integrated Value")

    # Manually define connections for this example (normally done by engine)
    # Connection: c1['out'] -> integ1['signal']
    # Connection: integ1['state'] -> plot1['in1']

    # Store blocks in a list (order doesn't *really* matter for placeholder)
    my_blocks = [c1, integ1, plot1]

    # --- Simulate (using placeholder loop) ---
    t_sim = 0.0
    dt_sim = 0.01
    t_stop_sim = 5.0

    # Data storage for manual connection simulation
    block_outputs = {} # Stores {'out': val} for each block instance

    print("\n--- Running Manual Simulation Example ---")
    while t_sim <= t_stop_sim:
        # Evaluate Constant
        c1_outputs = c1.update(t_sim, dt_sim, {})
        block_outputs[c1] = c1_outputs

        # Evaluate Integrator (using Constant's output)
        integ1_inputs = {'signal': block_outputs[c1]['out']}
        integ1_outputs = integ1.update(t_sim, dt_sim, integ1_inputs)
        block_outputs[integ1] = integ1_outputs

        # Evaluate Plot (using Integrator's output)
        plot1_inputs = {'in1': block_outputs[integ1]['state']}
        plot1_outputs = plot1.update(t_sim, dt_sim, plot1_inputs)
        # block_outputs[plot1] = plot1_outputs # Plot has no outputs

        t_sim += dt_sim

    print(f"--- Manual Simulation Finished at t={t_sim-dt_sim:.3f} ---")

    # Call end hooks
    print("--- Calling on_simulation_end hooks ---")
    for block in my_blocks:
        block.on_simulation_end()

    print("\n--- Example Complete ---")
    # Note: If you run this directly, it will show the plot generated by plot1.
