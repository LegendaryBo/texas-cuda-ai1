package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.comitstructs;

import java.util.*;

/**
 * Class representing single graph node it does some other things like building
 * graph from weight matrix, and calculating minimal spanning tree.
 * 
 * @author Kacper Gorski, Olgierd Humenczuk
 */
public class EvGraph {

  private ArrayList<EvEdge> nodes = new ArrayList<EvEdge>(); // edges adjecting

  // current Graph node

  private int nodes_number = 0; // number of nodes in whole graph containing

  // this Graph node

  private int label;


  /**
   * Contructs single graph node labeled with given int.<BR>
   * <BR>
   * 
   * @param label - label holded by created node
   */
  public EvGraph(int label) {
    this.label = label;
  }


  /**
   * Constructs whole full-edged graph from given table of int. <BR>
   * matrix[n][m] int table should contain weights of edge beetwen n and m graph
   * node and its size should be n x n.<BR>
   * <BR>
   * this object - single node of the graph. <BR>
   * <BR>
   * <B>NOTE</B>: only left bottom triangle without diagonal of the int table
   * is used to generate Graph, just like in the following example (only bold
   * values are used):<BR>
   * 2 6 2 1 8 <BR>
   * <B>2</b> 5 1 7 3 <BR>
   * <B>1 6</b> 8 5 2 <br>
   * <B>3 8 2</b> 6 8 <BR>
   * <B>2 5 7 8</b> 3 <BR>
   * 
   * @param matrix - square int table containing weights you want to build graph
   *        from
   */
  public EvGraph(int[][] matrix) {

    buildGraphFromMatrix(matrix);
  }


  /**
   * Build graph around this graph node from give int table.<BR>
   * <B>NOTE</B>: only left bottom triangle without diagonal of the int table
   * is used to generate Graph, just like in the following example (only bold
   * values are used):<BR>
   * 2 6 2 1 8 <BR>
   * <B>2</b> 5 1 7 3 <BR>
   * <B>1 6</b> 8 5 2 <br>
   * <B>3 8 2</b> 6 8 <BR>
   * <B>2 5 7 8</b> 3 <BR>
   * 
   * @param matrix - square int table containing weights you want to build graph
   *        from
   */
  private void buildGraphFromMatrix(int[][] matrix) {

    int height = nodes_number = matrix.length;
    int width = matrix[0].length;
    int h_it, w_it, w_stop = width - 1, h_stop = height;
    int i_it;

    EvGraph[] nodes = new EvGraph[height];
    nodes[0] = this;

    for (i_it = 1; i_it < nodes_number; i_it++)
      nodes[i_it] = new EvGraph(i_it);

    /** select random node as source node to find max spanning tree */
    // nodes[(int) (Math.random() * height)].weight = 0;
    // TODO - will it work??
    // nodes[0].weight = 0;
    EvEdge e = null;

    // building graph nodes and connecting it with edges
    for (w_it = 0; w_it < w_stop; w_it++) {
      for (h_it = w_it + 1; h_it < h_stop; h_it++) {
        e = new EvEdge(matrix[h_it][w_it], nodes[h_it], nodes[w_it]);
        nodes[w_it].addEdge(e);
        nodes[h_it].addEdge(e);
      }
    }

  }


  /**
   * Recursivly through adjectment graph nodes builds Hashmap containg Graph
   * objects as values and their labels as keys starting from current node. <BR>
   * Function add all not yet inserted Graph objects to given HashMap.
   * 
   * @param node_set
   */
  private void buildNodeSet(HashMap<Integer, EvGraph> node_set) {
    EvGraph v = null;

    for (EvEdge edge : nodes) {
      v = edge.getIncNode(this);
      if (!node_set.containsKey(v.label)) {
        node_set.put(v.label, v);
        v.buildNodeSet(node_set); // recursivly from every node
      }
    }
  }


  /**
   * Builds maximus spinning tree starting from current graph node.<BR>
   * Algorithm builds tree with every iteration of the loop by selecting the
   * highest edge coming from spinning tree and adds node adjecting to that
   * edge.
   */
  public EvTree getMaximumSpanningTree() {

    // temp datastructure for maximum spanning tree algorithm
    HashMap<Integer, EvGraph> g_node_set = new HashMap<Integer, EvGraph>();
    PriorityQueue<EvEdge> edge_queue = new PriorityQueue<EvEdge>(); // for
                                                                    // building
    // maximum
    // spinning tree

    // initializing Tree objects
    EvTree[] tree_nodes = new EvTree[nodes_number];
    for (int i_it = 0; i_it < nodes_number; i_it++) {
      tree_nodes[i_it] = new EvTree(i_it);
    }

    /** initialization edge queue */
    buildNodeSet(g_node_set);
    EvGraph start_node = g_node_set.get(0);
    g_node_set.remove(0);

    int node_size = start_node.nodes.size();
    for (int i = 0; i < node_size; i++) {
      edge_queue.add(start_node.nodes.get(i));
    }

    EvGraph v_node = null;
    EvGraph u_node = null;

    // loop building maximum spinning tree till no more nodes remains in queue
    while (!g_node_set.isEmpty()) {

      // get maximum edge
      EvEdge max_edge = edge_queue.poll();

      // get Graph node connected with selected maximum edge
      if (g_node_set.containsKey(max_edge.getFromNode().label)) {
        v_node = max_edge.getFromNode();
        u_node = max_edge.getToNode();
      } else {
        u_node = max_edge.getFromNode();
        v_node = max_edge.getToNode();
        if (!g_node_set.containsKey(max_edge.getToNode().label)) {
          v_node = null; // if we selected edge beetwen 2 nodes already in
                          // spinnig tree
        }
      }

      // if we didn't select edge beetwen 2 nodes already in spinnig tree
      if (v_node != null) {
        // set parent of Tree object with the same label as selected graph node
        // and set
        // it as adjectment Tree node
        tree_nodes[v_node.label].setParent(tree_nodes[u_node.label]);

        // remove from queue
        g_node_set.remove(v_node.label);

        // adding edges from selected Graph node at this loop step
        for (EvEdge edge : v_node.nodes) {
          if (g_node_set.containsKey(edge.getIncNode(v_node).label)
              && !edge_queue.contains(edge)) { // we only add not inserted
                                                // edges
            edge_queue.add(edge);
          }
        }
      }
    }

    EvTree parent = null;

    // build tree object
    for (int i_it = 0; i_it < nodes_number; i_it++) {
      parent = tree_nodes[i_it].getParent();
      if (parent != null) {
        parent.addChild(tree_nodes[i_it]);
      }
    }

    return tree_nodes[this.label];
  }


  /**
   * @return label of current Graph node
   */
  public int getLabel() {
    return label;
  }


  /**
   * @param label of current Graph node
   */
  public void setLabel(int label) {
    this.label = label;
  }


  /**
   * @return ArrayList of Edges objects adjecting to current Node
   */
  public ArrayList<EvEdge> getNodes() {
    return nodes;
  }


  /**
   * @param nodes ArrayList of Edges objects adjecting to current Node
   */
  public void setNodes(ArrayList<EvEdge> nodes) {
    this.nodes = nodes;
  }


  /**
   * @param e add Edge to list of edges adjecting to current node
   */
  public void addEdge(EvEdge e) {
    nodes.add(e);
  }

}
