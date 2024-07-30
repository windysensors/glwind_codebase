### wvavg.py ###
# Elliott Walker #
# 12 July 2024 #
# A demonstration of the 2 types of wind vector averaging #

import math

def mean(input_list: list) -> float:
  # input: a list of values to average
  # output: the arithmetic mean
  total = 0.
  for value in input_list:
    total += value
  result = total / len(input_list)
  return result

def unit_vector_average_direction(wind_directions: list) -> float:
  # input: a list of wind directions (in degrees)
  # output: the unit-vector-averaged mean wind direction [unweighted average wind direction] (in degrees)

  u_values = []
  v_values = []

  for direction in wind_directions:
    radians = math.radians(direction)
    u = math.sin(radians)
    v = math.cos(radians)
    u_values.append(u)
    v_values.append(v)

  mean_u = mean(u_values)
  mean_v = mean(v_values)

  result = math.degrees(math.atan2(mean_u, mean_v)) % 360
  return result

def true_vector_average_direction(wind_speeds: list, wind_directions: list) -> float:
  # input: two lists - one of wind speeds, and the other of corresponding wind directions (in degrees)
  # output: the true-vector averaged mean wind direction [average wind direction weighted by speeds] (in degrees)

  u_values = []
  v_values = []

  for speed, direction in zip(wind_speeds, wind_directions):
    radians = math.radians(direction)
    u = speed * math.sin(radians)
    v = speed * math.cos(radians)
    u_values.append(u)
    v_values.append(v)

  mean_u = mean(u_values)
  mean_v = mean(v_values)

  result = math.degrees(math.atan2(mean_u, mean_v)) % 360
  return result

if __name__ == '__main__':
  # example: generates 50 random wind vectors, computes the average direction with each method
  N = 50
  import random
  speeds = [4. * random.random() for _ in range(N)]
  directions = [random.normalvariate(mu=90., sigma=90.) % 360 for _ in range(N)]
  unit_avg_dir = unit_vector_average_direction(directions)
  tv_avg_dir = true_vector_average_direction(speeds, directions)
  print(f'Unit vector averaged mean wind direction: {unit_avg_dir:.3f} degrees')
  print(f'True-vector average mean wind direction: {tv_avg_dir:.3f} degrees')
