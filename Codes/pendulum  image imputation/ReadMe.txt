To simulate the experiment: Run "pendulum_image_imputation.py"

pendulum_image_imputation.py :  Pendulum image imputation
PendulumData.py : Ground truth state generation
GIN : GIN module; Subclass of tf.keras
GINTransitionCell : GIN layer (implemented in the paper)
GINSmoothCell : considering smoothing parameterization (implemented in the paper)
ImageGen : Noisy high dimensional observation generation
LayerNormalizer : Operates layer normalization

Files include comments for better readability