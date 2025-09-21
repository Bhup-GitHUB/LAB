numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print("Original list:", numbers)

squares_list = [x**2 for x in numbers]
print("Squares using list comprehension:", squares_list)

even_squares = [x**2 for x in numbers if x % 2 == 0]
print("Squares of even numbers:", even_squares)

cubes = [x**3 for x in range(1, 6)]
print("Cubes from 1 to 5:", cubes)

words = ["hello", "world", "python", "programming"]
word_lengths = [len(word) for word in words]
print("Word lengths:", word_lengths)

uppercase_words = [word.upper() for word in words]
print("Uppercase words:", uppercase_words)

print("\nDictionary comprehension:")

squares_dict = {x: x**2 for x in numbers}
print("Squares dictionary:", squares_dict)

even_squares_dict = {x: x**2 for x in numbers if x % 2 == 0}
print("Even squares dictionary:", even_squares_dict)

word_lengths_dict = {word: len(word) for word in words}
print("Word lengths dictionary:", word_lengths_dict)

numbers_dict = {x: "even" if x % 2 == 0 else "odd" for x in range(1, 11)}
print("Even/Odd dictionary:", numbers_dict)

fruits = ["apple", "banana", "orange", "grape"]
fruits_dict = {fruit: fruit.upper() for fruit in fruits}
print("Fruits dictionary:", fruits_dict)

print("\nNested comprehensions:")

matrix = [[i + j for j in range(3)] for i in range(3)]
print("Matrix:", matrix)

flat_matrix = [element for row in matrix for element in row]
print("Flattened matrix:", flat_matrix)

print("\nConditional comprehensions:")

filtered_squares = [x**2 for x in numbers if x > 5]
print("Squares of numbers > 5:", filtered_squares)

filtered_dict = {x: x**2 for x in numbers if x % 2 == 1}
print("Squares of odd numbers:", filtered_dict)

print("\nString comprehensions:")

text = "Hello World Python Programming"
words_from_text = [word for word in text.split()]
print("Words from text:", words_from_text)

word_lengths_from_text = {word: len(word) for word in text.split()}
print("Word lengths from text:", word_lengths_from_text)
