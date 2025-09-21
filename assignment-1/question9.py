def relu(x):
    if x > 0:
        return x
    else:
        return 0

def relu_ternary(x):
    return x if x > 0 else 0

print("ReLU function using conditional:")
test_values = [-5, -2, -1, 0, 1, 2, 5, 3.14, -3.14]

for x in test_values:
    result = relu(x)
    print(f"ReLU({x}) = {result}")

print("\nReLU function using ternary operator:")
for x in test_values:
    result = relu_ternary(x)
    print(f"ReLU({x}) = {result}")

print("\nReLU function using max:")
def relu_max(x):
    return max(0, x)

for x in test_values:
    result = relu_max(x)
    print(f"ReLU({x}) = {result}")

print("\nReLU function with list comprehension:")
relu_results = [relu(x) for x in test_values]
print("Input values:", test_values)
print("ReLU results:", relu_results)

print("\nReLU function for arrays:")
import numpy as np
arr = np.array([-3, -1, 0, 1, 3])
relu_arr = np.maximum(0, arr)
print("Array:", arr)
print("ReLU array:", relu_arr)

print("\nReLU function definition:")
print("ReLU(x) = max(0, x)")
print("ReLU(x) = x if x > 0 else 0")
