To simulate the experiment: Run "lorenz_state_estimation.py"

lorenz_state_estimation.py :  Lorenz state estimation
LorenzSysModel.py : Determines the dynamics of the system
GIN : GIN module; Subclass of tf.keras
GINTransitionCell : GIN layer (implemented in the paper)
GINSmoothCell : Performs smoothing (implemented in the paper)
parameters : Contains the parameters of the simulation


Files include comments for better readability