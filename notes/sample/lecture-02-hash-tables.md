# Lecture 2: Hash Tables

## The idea

A hash table maps keys to array slots. A hash function turns a key into an index, so lookup, insert and delete jump straight to one slot instead of scanning. Two keys can map to the same index; that is a collision.

## Collisions and separate chaining

With separate chaining, each slot holds a linked list of every entry that hashed there. Inserting adds to the list, and searching walks it until the key matches.

## Load factor and resizing

Load factor α = n/m is entries divided by slots. When α goes above 0.75, double the number of slots and rehash every entry. A single resize costs O(n), but because the array doubles each time, the cost of all inserts is spread out, so each insert is O(1) amortized.

## Running time

Average case: O(1) for insert, search and delete, when the load factor stays bounded and hashes spread keys evenly. Worst case: O(n), when every key lands in one chain.

## Lab 4

Lab 4 asks you to build a hash map with separate chaining in Java. It is due Friday of week 5 at 11:59 pm.
