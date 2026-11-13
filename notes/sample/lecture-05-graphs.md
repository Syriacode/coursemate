# Lecture 5: Graphs

## Representing a graph

A graph has V vertices and E edges. An adjacency list stores, for each vertex, a list of its neighbours and uses O(V + E) space. An adjacency matrix is a V × V table of 0s and 1s, uses O(V²) space, and answers "is there an edge from u to v?" in O(1).

## Breadth-first search

BFS starts at a source vertex and visits all vertices at distance 1, then distance 2, and so on, using a queue. Because it explores level by level, it finds the shortest path in an unweighted graph. Mark each vertex as visited when you add it to the queue so it is never added twice.

## Depth-first search

DFS follows one path as far as it can, then backs up and tries the next branch. It uses recursion or an explicit stack. It is the tool for detecting cycles, finding connected components, and ordering tasks that depend on each other.

## Running time

With an adjacency list, both BFS and DFS visit each vertex once and look at each edge once, so they run in O(V + E). With an adjacency matrix, finding a vertex's neighbours scans a whole row, which makes the same searches O(V²).
