# Lecture 3: Binary Search Trees

## The BST property

In a binary search tree, keys in a node's left subtree are smaller than the node, and keys in its right subtree are larger. Each comparison discards one whole subtree.

## Search and insert

Search goes left when the key is smaller, right when larger, and stops when equal. Insert follows the same path and adds a new leaf where it ends.

## Delete

Three cases, depending on the node:

1. Leaf: remove it.
2. One child: splice the child into its place.
3. Two children: copy the in-order successor (the smallest key in the right subtree) into the node, then delete that successor, which has at most one child.

## Height and running time

Height h is the longest root-to-leaf path, in edges. Search, insert and delete cost O(h): O(log n) when the tree is balanced, but O(n) when sorted inserts turn it into a list.

## Traversals

Pre-order visits node, left, right. In-order visits left, node, right, and on a BST it yields keys in sorted order. Post-order visits left, right, node.
