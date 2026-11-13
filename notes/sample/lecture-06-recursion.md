# Lecture 6: Recursion and Dynamic Programming

## Recursion and base cases

A recursive function solves a problem by calling itself on a smaller version of the same problem. Every recursive function needs a base case, an input it answers directly, and each call must move closer to that base case. Without a base case the calls never stop and the program runs out of stack space.

## Memoization

The naive recursive Fibonacci recomputes the same values again and again and takes exponential time. Memoization stores each result in a table the first time it is computed and looks it up afterwards. With a memo, Fibonacci of n makes only O(n) distinct calls.

## Dynamic programming

Dynamic programming solves a problem by combining answers to overlapping subproblems. A problem fits when it has optimal substructure, meaning the best answer is built from best answers to smaller pieces. Bottom-up dynamic programming fills a table from the smallest subproblems upward, with no recursion at all.
