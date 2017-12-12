import numpy as np
from sklearn import metrics

def get_squared_distances(points_array, other_points_array):
  """ Get square distances from points to other points
  """
  get_distances_to_sample = lambda other_point: np.sum((other_point - points_array) ** 2, axis=1).reshape(1, -1)
  squared_distances = np.array(map(get_distances_to_sample, other_points_array)).reshape(other_points_array.shape[0], -1)
  return squared_distances

def get_covariance(points_array, parameters, other_points_array=[]):
  """ Get covariance matrix
  """
  if other_points_array == []:
    other_points_array = points_array
    single_sample = True
  else:
    single_sample = False

  squared_distances_array = get_squared_distances(points_array, other_points_array)
  # covariance_matrix = (np.exp(-parameters['theta'] * squared_distances_array) + 
  #                      parameters['noise_variance'] * np.eye(points_array.shape[0]))
  covariance_matrix = np.exp(-parameters['theta'] * squared_distances_array)
  if single_sample and ('noise_variance' in parameters):
    sample_size = np.shape(points_array)[0]
    covariance_matrix = covariance_matrix + parameters['noise_variance'] ** 0.5 * np.eye(sample_size)
  return covariance_matrix


def get_covariance_matern_12(points_array, parameters, other_points_array=[]):
  """ Get covariance matrix
  """
  if other_points_array == []:
    other_points_array = points_array
    single_sample = True
  else:
    single_sample = False

  # squared_distances_array = get_squared_distances(points_array, other_points_array)
  squared_distances_array = metrics.euclidean_distances(points_array, other_points_array, squared=True)
  # covariance_matrix = (np.exp(-parameters['theta'] * squared_distances_array) + 
  #                      parameters['noise_variance'] * np.eye(points_array.shape[0]))
  covariance_matrix = np.exp(-parameters['theta'] * squared_distances_array ** 0.5)
  if single_sample and ('noise_variance' in parameters):
    sample_size = np.shape(points_array)[0]
    covariance_matrix = covariance_matrix + parameters['noise_variance'] ** 0.5 * np.eye(sample_size)
  return covariance_matrix

def get_covariance_matern_32(points_array, parameters, other_points_array=[]):
  """ Get covariance matrix
  """
  if other_points_array == []:
    other_points_array = points_array
    single_sample = True
  else:
    single_sample = False

  # squared_distances_array = get_squared_distances(points_array, other_points_array)
  squared_distances_array = metrics.euclidean_distances(points_array, other_points_array, squared=True)
  # covariance_matrix = (np.exp(-parameters['theta'] * squared_distances_array) + 
  #                      parameters['noise_variance'] * np.eye(points_array.shape[0]))
  covariance_matrix = ((1 + 3**0.5 * parameters['theta'] * squared_distances_array ** 0.5) * 
                       np.exp(-3**0.5 * parameters['theta'] * squared_distances_array ** 0.5))
  if single_sample and ('noise_variance' in parameters):
    sample_size = np.shape(points_array)[0]
    covariance_matrix = covariance_matrix + parameters['noise_variance'] ** 0.5 * np.eye(sample_size)
  return covariance_matrix

def get_covariance_matern_52(points_array, parameters, other_points_array=[]):
  """ Get covariance matrix
  """
  if other_points_array == []:
    other_points_array = points_array
    single_sample = True
  else:
    single_sample = False

  squared_distances_array = get_squared_distances(points_array, other_points_array)
  # covariance_matrix = (np.exp(-parameters['theta'] * squared_distances_array) + 
  #                      parameters['noise_variance'] * np.eye(points_array.shape[0]))
  covariance_matrix = ((1 + 5**0.5 * parameters['theta'] ** 0.5 * squared_distances_array ** 0.5 +
                        5 / 3. * parameters['theta'] * squared_distances_array) * 
                       np.exp(-5**0.5 * parameters['theta'] ** 0.5 * squared_distances_array ** 0.5))
  if single_sample and ('noise_variance' in parameters):
    sample_size = np.shape(points_array)[0]
    covariance_matrix = covariance_matrix + parameters['noise_variance'] ** 0.5 * np.eye(sample_size)
  return covariance_matrix

def calculate_regret(covariance_matrix, noise_variance):
  sample_size = np.shape(covariance_matrix)[0]
  shifted_eigenvalues = np.linalg.eigvalsh(covariance_matrix / noise_variance + np.eye(sample_size))
  return np.sum(np.log(shifted_eigenvalues))

def get_regret_value(new_points_array, points_array, parameters):
  return np.array(map(lambda new_point: calculate_regret(get_covariance(np.vstack((points_array, [new_point])), parameters), 
                                                                         parameters['noise_variance']), new_points_array))

def get_accuracy_evaluation(points_array, new_points_array, parameters, cholesky_covariance_matrix):
  """ Get accuracy evaluation for GP model
  """
  new_covariance = get_covariance(new_points_array, parameters, points_array)
  alpha = np.linalg.solve(cholesky_covariance_matrix, new_covariance).transpose()
  ae_values = np.ones((new_points_array.shape[0], )) + parameters['noise_variance'] - np.sum(alpha * alpha, axis=1)

  return ae_values