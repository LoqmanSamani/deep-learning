def integral(min_, max_, func_, n, method):
    """Approximate the area under the curve of func_ from min_ to max_ using n rectangles."""
    delta_x = (max_ - min_) / n
    area = 0
    for i in range(1, n + 1):
      if method == "left":
        x = min_ + (i - 1) * delta_x  # left Riemann sum
        area += func_(x, delta_x)
      elif method == "right":
        x = min_ + i * delta_x  # right Riemann sum
        area += func_(x, delta_x)
      elif method == "middle":
        x = min_ + (i - 0.5) * delta_x  # middle Riemann sum
        area += func_(x, delta_x)
    return area

def x_power(i, delta_x):
  return i**2 * delta_x

print(integral(0, 1, x_power, 5, "left"))
print(integral(0, 1, x_power, 5, "right"))
print(integral(0, 1, x_power, 5, "middle"))  

"""
0.24000000000000005
0.44000000000000006
0.33000000000000007
"""

print(integral(0, 1, x_power, 100, "left"))
print(integral(0, 1, x_power, 100, "right"))
print(integral(0, 1, x_power, 100, "middle")) 

"""
0.32835000000000014
0.33835000000000015
0.33332500000000004
"""
