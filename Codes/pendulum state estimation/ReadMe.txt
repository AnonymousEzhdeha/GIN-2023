To simulate the experiment: Run "pendulum_state_estimation.py"

pendulum_state_estimation.py :  Pendulum state estimation
PendulumData.py : Ground truth state generation
GIN : GIN module; Subclass of tf.keras
GINTransitionCell : GIN layer (implemented in the paper)
GINSmoothCell : considering smoothing parameterization (implemented in the paper)
ImageGen : Noisy high dimensional observation generation
LayerNormalizer : Operates layer normalization

Files include comments for better readability