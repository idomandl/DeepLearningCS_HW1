from function.tanh import Tanh
import numpy as np
from algorithm.jacobian_transposed_test import Jacobian_Transposed_Test

jac_t_test = Jacobian_Transposed_Test(Tanh())
results = [jac_t_test(np.random.rand(1, 10), np.random.rand(10, 5) * 0.01) for _ in range(1000)]
print(max(results))
