# filename: my_model.py
# Bouncing Ball Simulation
# Requires updates to sim_blocks.py (New: Memory, Bounce blocks; Modified: Integrator)

# Required packages: matplotlib, numpy

# --- BlockSim Imports (Assumes updated sim_blocks.py exists) ---
# Need to ensure these classes exist and Integrator is modified
from sim_blocks import Constant, Integrator, Plot, Memory, Bounce, simulate

# --- Simulation Parameters ---
GRAVITY = -32.2  # ft/s^2
INITIAL_HEIGHT = 5.0 # ft
INITIAL_VELOCITY = 0.0 # ft/s
COR = 0.7 # Coefficient of Restitution (0 to 1)
GROUND_LEVEL = 0.0 # ft

# --- Model Definition Function (Not directly executed by engine) ---
# This function's code snippets primarily serve as configuration storage
# for the GUI and parameter extraction during 'on_run'.
def my_model(t, dt):
    # The engine executes blocks based on connections, not this code order.
    # This code is mainly for block instantiation configuration.

    # Constant Acceleration
    g_const = Constant(value=GRAVITY)

    # Velocity Integrator - MODIFIED: needs 'override_in' port
    # Parameter: initial=INITIAL_VELOCITY
    velocity_integ = Integrator(initial=INITIAL_VELOCITY)

    # Position Integrator
    # Parameter: initial=INITIAL_HEIGHT
    position_integ = Integrator(initial=INITIAL_HEIGHT)

    # Memory Block (Unit Delay) - NEW BLOCK
    pos_memory = Memory(initial=INITIAL_HEIGHT)
     # Initial value matches position

    # Bounce Logic Block - NEW BLOCK
    # Parameters: COR=COR, ground_level=GROUND_LEVEL
    bounce_logic = Bounce(COR=COR, ground_level=GROUND_LEVEL)

    # Plot Position
    plot_pos = Plot(title="Bouncing Ball Position", ylabel="Height (ft)")

    # Plot Velocity
    plot_vel = Plot(title="Bouncing Ball Velocity", ylabel="Velocity (ft/s)")

    # The return value of this function is ignored by the simulation engine.
    return None
     # Or perhaps the final position if needed elsewhere

    # --- Main execution block (Only runs if script is executed directly) ---
    # This section will NOT be used when running from the BlockSim GUI's "Run" menu.
    # The GUI's "Run" button now builds instances and uses the engine directly.
if __name__ == "__main__":
    print("This script defines a bouncing ball model for BlockSim.")
    print("To run the simulation visually, load this file in block_gui.py and press Run.")
    print("\nNOTE: Running this requires updates to sim_blocks.py:")
    print("  - Add 'Memory' block class.")
    print("  - Add 'Bounce' block class.")
    print("  - Modify 'Integrator' to accept 'override_in' port and logic.")

    # Example of how you *could* run it programmatically if blocks were defined
    # block_map = {
    #     1: Constant(value=GRAVITY),
    #     2: Integrator(initial=INITIAL_VELOCITY), # Needs modified Integrator
    #     3: Integrator(initial=INITIAL_HEIGHT),
    #     4: Memory(initial=INITIAL_HEIGHT),      # Needs Memory block
    #     5: Bounce(COR=COR, ground_level=GROUND_LEVEL), # Needs Bounce block
    #     6: Plot(title="Bouncing Ball Position", ylabel="Height (ft)"),
    #     7: Plot(title="Bouncing Ball Velocity", ylabel="Velocity (ft/s)")
    # }
    # connections = [ # From JSON below
    #     [1, "out", 2, "signal"], [2, "state", 3, "signal"],
    #     [2, "state", 5, "vel_in"], [3, "state", 4, "in"],
    #     [3, "state", 6, "in1"], [4, "out", 5, "pos_in"],
    #     [5, "vel_override", 2, "override_in"], [2, "state", 7, "in1"]
    # ]
    # print("\nAttempting simulation run (will fail without updated sim_blocks)...")
    # simulate(block_map, connections, t_stop=4.0, dt=0.01)
