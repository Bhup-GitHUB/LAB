def relu(x):
    if x >= 0:
        return x
    else:
        return 0
print(relu(-3))
print(relu(5))

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
print(factorial(5))

with open("file.txt", "r") as f:
    words = f.read().split()
unique_words = sorted(set(words))
print(unique_words)