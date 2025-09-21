def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

print("Prime numbers less than 20:")
primes = []
for i in range(2, 20):
    if is_prime(i):
        primes.append(i)
        print(i)

print("List of primes:", primes)

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print("\nFactorial using recursion:")
test_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for num in test_numbers:
    result = factorial(num)
    print(f"{num}! = {result}")

def find_unique_words(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            words = content.lower().split()
            unique_words = sorted(set(words))
            return unique_words
    except FileNotFoundError:
        print(f"File '{filename}' not found. Creating a sample file...")
        sample_text = """This is a sample text file.
        It contains multiple words and some repeated words.
        The program should find unique words in alphabetical order.
        This is another sentence with more words."""
        
        with open(filename, 'w') as file:
            file.write(sample_text)
        
        with open(filename, 'r') as file:
            content = file.read()
            words = content.lower().split()
            unique_words = sorted(set(words))
            return unique_words

print("\nUnique words in alphabetical order:")
filename = "sample.txt"
unique_words = find_unique_words(filename)
print("Unique words:", unique_words)

print("\nAll three parts together:")
print("1. Prime numbers less than 20:", primes)
print("2. Factorial of 5:", factorial(5))
print("3. Unique words from file:", len(unique_words), "words found")
