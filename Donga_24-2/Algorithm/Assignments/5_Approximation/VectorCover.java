import java.io.*;
import java.util.*;
import java.util.LinkedList;

// This class represents an undirected graph using adjacency list
class VectorCover
{
    private int V;   // No. of vertices
    private LinkedList<Integer> adj[]; // Array of lists for Adjacency List Representation

    // Constructor
    VectorCover(int v)
    {
        V = v;
        adj = new LinkedList[v];
        for (int i = 0; i < v; ++i)
            adj[i] = new LinkedList();
    }

    // Function to add an edge into the graph
    void addEdge(int v, int w)
    {
        adj[v].add(w);  // Add w to v's list.
        adj[w].add(v);  // Graph is undirected
    }

    // The function to print the vertex cover using a greedy approximation
    void printVertexCover()
    {
        // Initialize all vertices as not visited.
        boolean visited[] = new boolean[V];
        Arrays.fill(visited, false);  // Mark all vertices as not visited

        // Store the vertex cover set
        Set<Integer> vertexCover = new HashSet<>();

        // Traverse all edges and select an edge when both vertices are not visited
        for (int u = 0; u < V; u++)
        {
            Iterator<Integer> i = adj[u].iterator();
            while (i.hasNext())
            {
                int v = i.next();

                // If neither u nor v is visited, add both to the vertex cover
                if (!visited[u] && !visited[v])
                {
                    // Add the vertices (u, v) to the result set
                    vertexCover.add(u);
                    vertexCover.add(v);

                    // Mark both vertices as visited
                    visited[u] = true;
                    visited[v] = true;
                    break;  // Move to the next edge
                }
            }
        }

        // Print the vertex cover
        System.out.println("Vertex Cover: " + vertexCover);
    }

    // Driver method
    public static void main(String args[])
    {
        // Create a graph with 7 vertices (you can modify this as needed)
        VectorCover g = new VectorCover(7);
        g.addEdge(0, 1);
        g.addEdge(0, 2);
        g.addEdge(1, 3);
        g.addEdge(3, 4);
        g.addEdge(4, 5);
        g.addEdge(5, 6);

        // Print the vertex cover
        g.printVertexCover();
    }
}
