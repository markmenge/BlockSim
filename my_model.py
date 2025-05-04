# my_model.py

# Required packages:
#   pip install matplotlib numpy redbaron

from sim_blocks import simulate, Integrator, Scope

# Instantiate blocks
integrator = Integrator(initial=0.0)
scope = Scope()

def my_model(t, dt):
    # blockid=1;x=190;y=76
    c1 = Constant(value=0.1)
    # blockid=2;x=367;y=118
    i2 = Integrator(initial=5.0)
    # blockid=3;x=543;y=147
    p3 = Plot()
    return y

if __name__ == "__main__":
    # Run the simulation for 5 seconds with a 10 ms timestep
    simulate(my_model, t_stop=5.0, dt=0.01, scope=scope)
    # Plot the results
    scope.plot(title="BlockSim Output", ylabel="y(t)")
