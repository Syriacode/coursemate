# Lecture 4: Sorting

## Why sorting matters

Sorted data makes many tasks easier: binary search finds a key in O(log n), duplicates sit next to each other, and merging two sorted lists is one pass. Sorting costs time up front, so it pays off when you search or scan the data many times.

## Insertion sort

Insertion sort keeps a sorted prefix and inserts the next element into its correct place by shifting larger elements right. It takes O(n²) time in the worst case but O(n) on an already sorted array, and it is fast on small inputs.

## Merge sort

Merge sort splits the array in half, sorts each half recursively, and merges the two sorted halves. Merging takes O(n) and there are log n levels of splitting, so it runs in O(n log n) in every case. It needs O(n) extra space for the merge.

## Quicksort

Quicksort picks a pivot, partitions the array into smaller and larger elements, and recurses on both sides. It averages O(n log n) and sorts in place, but a bad pivot choice, such as always taking the first element of a sorted array, gives O(n²). Picking a random pivot makes the bad case very unlikely.

## Stability

A sort is stable if equal keys keep their original relative order. Insertion sort and merge sort are stable; the usual in-place quicksort is not. Stability matters when you sort by one field after another, for example by name and then by year.
