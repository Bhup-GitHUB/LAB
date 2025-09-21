mixed_list = [1, "hello", 3.14, True, [1, 2, 3], {"key": "value"}]

print("Original list:", mixed_list)
print("Data types in list:")
for i, item in enumerate(mixed_list):
    print(f"Index {i}: {item} ({type(item).__name__})")

print("\nAccessing elements:")
print("First element:", mixed_list[0])
print("Second element:", mixed_list[1])
print("Last element:", mixed_list[-1])

print("\nSlicing:")
print("First 3 elements:", mixed_list[:3])
print("Elements from index 2 to end:", mixed_list[2:])
print("Elements from index 1 to 4:", mixed_list[1:5])

print("\nAppending elements:")
mixed_list.append("new item")
print("After append:", mixed_list)

mixed_list.append(42)
print("After append number:", mixed_list)

print("\nRemoving elements:")
mixed_list.remove("hello")
print("After remove 'hello':", mixed_list)

removed = mixed_list.pop()
print("Popped element:", removed)
print("After pop:", mixed_list)

print("\nList operations:")
numbers = [1, 2, 3, 4, 5]
print("Numbers list:", numbers)
print("Sum:", sum(numbers))
print("Length:", len(numbers))
print("Max:", max(numbers))
print("Min:", min(numbers))

numbers.extend([6, 7, 8])
print("After extend:", numbers)

numbers.insert(2, 99)
print("After insert 99 at index 2:", numbers)

print("\nList comprehension:")
squares = [x**2 for x in range(1, 6)]
print("Squares:", squares)
