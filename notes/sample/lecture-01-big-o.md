# Lecture 1: Big-O Notation

## Why measure growth

Measured seconds depend on the machine, compiler and input, so they don't compare algorithms fairly. Instead we count how the number of steps grows as the input size n grows. Two algorithms can look alike for small n but differ enormously at n = 1,000,000. Growth rate decides which one scales.

## Formal definition

We say f(n) is O(g(n)) if there are constants c > 0 and n₀ such that f(n) ≤ c·g(n) for all n ≥ n₀. Big-O is an upper bound that ignores constant factors, so 3n² + 5n is O(n²).

## Common classes

- O(1): constant. Reading the first element of an array.
- O(log n): logarithmic. Binary search on a sorted array.
- O(n): linear. Scanning a list once to find the maximum.
- O(n log n): linearithmic. Merge sort.
- O(n²): quadratic. Comparing every pair of elements.
- O(2ⁿ): exponential. Generating every subset of a set.

## Best, average and worst case

Cases describe which input we run on, not a bound. For linear search, the best case is O(1) when the target is first, the worst case is O(n) when it's absent, and the average case, with the target equally likely anywhere, is about n/2 comparisons, still O(n).
