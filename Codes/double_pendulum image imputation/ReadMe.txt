To simulate the experiment: Run "double_pendulum_image_imputation.py"

double_pendulum_image_imputation.py : Double pendulum image imputation
DoublePendulum.py : Ground truth state generation
GIN : GIN module; Subclass of tf.keras
GINTransitionCell : GIN layer (implemented in the paper)
GINSmoothCell : considering smoothing parameterization (implemented in the paper)
ImageGen : Noisy high dimensional observation generation
LayerNormalizer : Operates layer normalization

Files include comments for better readability.