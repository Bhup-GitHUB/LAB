mixed_set = {1, "hello", 3.14, True, (1, 2, 3)}

print("Original set:", mixed_set)
print("Data types in set:")
for item in mixed_set:
    print(f"{item} ({type(item).__name__})")

print("\nChecking for elements:")
print("1 in set:", 1 in mixed_set)
print("'hello' in set:", "hello" in mixed_set)
print("'world' in set:", "world" in mixed_set)
print("True in set:", True in mixed_set)

print("\nAdding elements:")
mixed_set.add("world")
print("After adding 'world':", mixed_set)

mixed_set.add(42)
print("After adding 42:", mixed_set)

mixed_set.update([5, 6, 7])
print("After updating with [5, 6, 7]:", mixed_set)

print("\nRemoving elements:")
mixed_set.remove("hello")
print("After removing 'hello':", mixed_set)

mixed_set.discard("world")
print("After discarding 'world':", mixed_set)

removed = mixed_set.pop()
print("Popped element:", removed)
print("After pop:", mixed_set)

print("\nSet operations:")
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

print("Set 1:", set1)
print("Set 2:", set2)

print("Union:", set1.union(set2))
print("Intersection:", set1.intersection(set2))
print("Difference (set1 - set2):", set1.difference(set2))
print("Difference (set2 - set1):", set2.difference(set1))
print("Symmetric difference:", set1.symmetric_difference(set2))

print("\nSet methods:")
print("Length:", len(mixed_set))
print("Is subset (set1 subset of {1,2,3,4,5,6}):", set1.issubset({1,2,3,4,5,6}))
print("Is superset (set1 superset of {1,2,3}):", set1.issuperset({1,2,3}))

print("\nSet comprehension:")
squares_set = {x**2 for x in range(1, 6)}
print("Squares set:", squares_set)
