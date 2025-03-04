import numpy as np
#length_sale, and sigma
class SquaredExponentialKernel:
    def __init__(self, sigma_f: float = 1, length: float = 1):
        self.sigma_f = sigma_f
        self.length = length

    def __call__(self, argument_1: np.array, argument_2: np.array) -> float:
        return float(self.sigma_f *
                     np.exp(-(np.linalg.norm(argument_1 - argument_2)**2) /
                            (2 * self.length**2)))


# Helper function to calculate the respective covariance matrices
def cov_matrix(x1, x2, cov_function) -> np.array:
  try:
    return np.array([[cov_function(a, b) for a in x1] for b in x2])
  except TypeError as e:
    print("Using Matern")
    return np.array([[cov_function(a.reshape(-1,1), b.reshape(-1,1)) for a in x1] for b in x2])

class GPR:
    def __init__(self,
                 data_x: np.array,
                 data_y: np.array,
                 covariance_function=SquaredExponentialKernel(),
                 white_noise_sigma: float = 0):

        self.noise = white_noise_sigma
        self.data_x = data_x
        self.data_y = data_y
        self.covariance_function = covariance_function

        # Store the inverse of covariance matrix of input (+ machine epsilon on diagonal) since it is needed for every prediction
        # machine epsilon to prevent non-invertible covariance matrix since we are dealing with matrix algebra
        #K^{-1}
        self._inverse_of_covariance_matrix_of_input = np.linalg.inv(
            cov_matrix(data_x, data_x, covariance_function) +
            (3e-7 + self.noise) * np.identity(len(self.data_x)))

        self._memory = None

    # function to predict output at new input values. Store the mean and covariance matrix in memory.
    def predict(self, at_values: np.array) -> np.array:
        #K_* = k(x,x*)
        k_lower_star = cov_matrix(self.data_x, at_values,
                                  self.covariance_function)
        #K_** = l(x_*,x_*)
        k_lower_dstar = cov_matrix(at_values, at_values,
                                   self.covariance_function)
        # Mean K_*K^{-1}y.
        mean_at_values = np.dot(
            k_lower_star,
            np.dot(self.data_y,
                   self._inverse_of_covariance_matrix_of_input.T).T).flatten()

        # Covariance K_**-K_*K^{-1}K_*^T.
        cov_at_values = k_lower_dstar - \
            np.dot(k_lower_star, np.dot(
                self._inverse_of_covariance_matrix_of_input, k_lower_star.T))

        # Adding value larger than machine epsilon to ensure positive semi definite
        cov_at_values = cov_at_values + 3e-7 * np.ones(
            np.shape(cov_at_values)[0])

        var_at_values = np.diag(cov_at_values)

        self._memory = {
            'mean': mean_at_values,
            'covariance_matrix': cov_at_values,
            'variance': var_at_values
        }
        return mean_at_values
