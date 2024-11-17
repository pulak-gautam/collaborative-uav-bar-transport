# Collaborative UAV Bar Transport
Recreating centralized linear MPC results from the paper <a href="https://ieeexplore.ieee.org/document/9811726">"Decentralized Model Predictive Control for Equilibrium-based
Collaborative UAV Bar Transportation"</a> ICRA'22

## Prerequisites
``pip install numpy matplotlib scipy casadi``

## Running the Code

Following lines on code execution defines initial state vector and target setpoint. Modify these with your desired values. You may also change value of ``thetal`` and ``thetaf`` at eqb, by passing your desired values.
```python
  x0, u0 = get_eqb(pb = [1, 1, 2], thetal=thetal, thetaf=thetaf) #ref variables
  xinit, uinit = get_eqb(thetal=thetal, thetaf=thetaf) #initial variables
# get_eqb(pb, yawb, yaw1, yaw2, thetaf, thetal) -> x, u
```
Default value takes initial position of bar ``pb`` to be ``[0, 0, 2]`` and all yaws as zero, and both ``thetal`` and ``thetaf`` to be zeros. 

Once you have defined your desired initial state vector and control input (``xinit`` and ``uinit``), and target setpoint (``x0`` and ``u0``), you can simply run the python file to output controller results which includes 3D plot of trajectory and graphs showing tracking of state vector. 
```python
python controller.py
```
## Implementation Details
The single python implements the entire dynamics, linearization, discretization and finally a linear MPC solver using ``ipopt`` solver using ``casadi`` symbolic library.

