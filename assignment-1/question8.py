def print_sign(number):
    if number > 0:
        print(f"{number} is positive")
    elif number < 0:
        print(f"{number} is negative")
    else:
        print(f"{number} is zero")

numbers = [5, -3, 0, 10, -7.5, 0.0, -100]

print("Checking sign of numbers:")
for num in numbers:
    print_sign(num)

print("\nUsing conditional expressions:")
for num in numbers:
    sign = "positive" if num > 0 else "negative" if num < 0 else "zero"
    print(f"{num} is {sign}")

print("\nUsing abs() function:")
for num in numbers:
    if num != 0:
        print(f"Absolute value of {num} is {abs(num)}")

print("\nSign function using math.copysign:")
import math
for num in numbers:
    if num != 0:
        sign = 1 if math.copysign(1, num) > 0 else -1
        print(f"Sign of {num} is {sign}")

print("\nCustom sign function:")
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

for num in numbers:
    print(f"Sign of {num} is {sign(num)}")

print("\nTesting with float numbers:")
float_numbers = [3.14, -2.718, 0.0, -0.5, 100.0]
for num in float_numbers:
    print_sign(num)
