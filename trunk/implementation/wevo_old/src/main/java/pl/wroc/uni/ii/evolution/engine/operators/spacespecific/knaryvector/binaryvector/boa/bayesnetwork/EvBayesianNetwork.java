package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.bayesnetwork;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.utils.EvBinary;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.utils.EvTriple;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Simple implementation of bayesian network.
 * Each vertex represents single gene, each edge tells, that there is and
 * dependency beetwen vertexes connected by the edge
 * 
 * @author Marcin Golebiowski, Jarek Fuks, Zbigniew Nazimek
 */
public class EvBayesianNetwork {

  /**
   * number of variables (also vertexes in network).
   */
  private int vertex_num;

  /**
   * max. number of parent per vertex in network
   */
  private int max_parents;

  /**
   * current number of edges in whole graph.
   */
  private int number_of_edges;

  /**
   * Vertes of the network. Each vertex has index corresponding to
   * position in the table
   */
  private EvBayesianNetworkNode[] nodes;

  /**
   * Tells if probabilities in network has been already generated using
   * population.
   */
  private boolean generated_probabilities = false;

  /**
   * connected[i][j] is true when there is a an edge (one or more) heading from
   * vertex i to j.
   */
  private boolean[][] connected = null;


  /**
   * Constructor.
   * 
   * @param number_of_var -- number of variables
   * @param number_of_par -- max number of parents
   */
  public EvBayesianNetwork(final int number_of_var, final int number_of_par) {
    this.vertex_num = number_of_var;
    this.max_parents = number_of_par;
    this.nodes = new EvBayesianNetworkNode[vertex_num];

    for (int i = 0; i < vertex_num; i++) {
      nodes[i] = new EvBayesianNetworkNode(i, vertex_num);
    }

    connected = new boolean[vertex_num][vertex_num];

    for (int i = 0; i < vertex_num; i++) {
      for (int j = 0; j < vertex_num; j++) {
        if (i != j) {
          connected[i][j] = false;
        } else {
          connected[i][j] = true;
        }
      }
    }
  }


  /**
   * @return number of variables
   */
  public int getN() {
    return vertex_num;
  }


  /**
   * @return output degree constrain
   */
  public int getK() {
    return max_parents;
  }


  /**
   * @param child_index index of child
   * @return indexes of parent of child
   */
  public int[] getParentsIndexes(final int child_index) {
    EvBayesianNetworkNode[] parents = nodes[child_index].getParents();
    int[] parent_indexes = new int[parents.length];
    int i = 0;
    for (EvBayesianNetworkNode node : parents) {
      parent_indexes[i++] = node.getIndex();
    }
    return parent_indexes;
  }


  /**
   * @param parent_index - parent_index whose child will be returned
   * @return indexes of parent of child specified by given index
   */
  public int[] getChildIndexes(final int parent_index) {
    EvBayesianNetworkNode[] parents = nodes[parent_index].getChildren();
    int[] parent_indexes = new int[parents.length];
    int i = 0;
    for (EvBayesianNetworkNode node : parents) {
      parent_indexes[i++] = node.getIndex();
    }
    return parent_indexes;
  }


  /**
   * Returns true if adding edge from vertex j to i is legal.
   * 
   * @param j - starting vertex
   * @param i - ending vertex
   * @return false if one of the following occur:<br>
   * - i == j<br>
   * - i has already maximum number of parents
   * - j is already parent of i
   * - there is already connection from i to j (adding edge will lead into
   * cycle)
   * 
   */
  public boolean legalEdge(final int i, final int j) {
    return (i != j) && (nodes[i].getParentCount() < max_parents)
        && (!nodes[i].hasParent(nodes[j])) && (!connected[j][i]);
  }


  /**
   * Add edge starting from node j to i.
   * 
   * @param i - child node
   * @param j - parent node
   * @return true if operation is legal, false otherwise
   */
  public boolean addEdge(final int i, final int j) {
    if (!legalEdge(i, j)) {
      return false;
    }
    nodes[i].addParent(nodes[j]);
    nodes[j].addChild(nodes[i]);
    connected[i][j] = true;

    for (int k = 0; k < vertex_num; k++) {
      if (connected[k][i]) {
        for (int l = 0; l < vertex_num; l++) {
          if ((l != k) && (connected[j][l])) {
            connected[k][l] = true;
          }
        }
        connected[k][j] = true;
      }

    }

    number_of_edges++;
    generated_probabilities = false;

    return true;
  }

  /**
   * Removes specified edge.
   * 
   * @param i - child vertex
   * @param j - parent vertex
   * @return true if there was such edge, false otherwise
   */
  public boolean removeEdge(final int i, final int j) {
    if (!nodes[j].hasChild(nodes[i])) {
      return false;
    }

    number_of_edges--;
    nodes[i].removeParent(nodes[j]);
    nodes[j].removeChild(nodes[i]);
    generated_probabilities = false;
    return true;
  }

  /**
   * 
   * @param index - vertex index
   * @return number of parents of specified vertex 
   */
  public int getParentsIndexesCount(final int index) {
    return nodes[index].getParentCount();
  }

  /**
   * 
   * @return total number of vertexes in the network
   */
  public int getNumberOfEdges() {
    return number_of_edges;
  }


  /**
   * Generates m individual using this bayesian network.
   * Each edges notifies that there is an dependency childreen and parental
   * gene
   * 
   * @param population whose individuals will be used to update probability
   * values on network's edges
   * @param m - number of individuals to be generates
   * @return new population containing m generates individuals
   */
  public EvBinaryVectorIndividual[] generate(
      final EvBinaryVectorIndividual[] population, final int m) {

    EvObjectiveFunction<EvBinaryVectorIndividual> fun =
        population[0].getObjectiveFunction();
    EvBinaryVectorIndividual[] result = new EvBinaryVectorIndividual[m];
    int d = population[0].getDimension();

    boolean[] computed = new boolean[this.vertex_num];

    for (int i = 0; i < vertex_num; i++) {
      computed[i] = false;
    }

    for (int i = 0; i < m; i++) {
      result[i] = new EvBinaryVectorIndividual(d);
      result[i].setObjectiveFunction(fun);
    }

    if (!generated_probabilities) {
      generatePropabilities(population);
      generated_probabilities = true;
    }

    int nodes_size = nodes.length;

    for (int i = 0; i < nodes_size; i++) {
      nodes[i].generate(result, computed);
    }

    return result;
  }

  /**
   * Generates probabilities of gene 0 or 1 on each vertex in the network 
   * using given population.
   * 
   * @param population - sample population
   */
  private void generatePropabilities(
      final EvBinaryVectorIndividual[] population) {

    final double initialProbability = 0.5d;
    
    int[] parents_indexes;
    EvBayesianNetworkNode node;

    for (int n = 0; n < nodes.length; n++) {
      node = nodes[n];
      EvBayesianNetworkNode[] parents = node.getParents();

      parents_indexes = new int[parents.length];

      for (int j = 0; j < parents_indexes.length; j++) {
        parents_indexes[j] = parents[j].getIndex();
      }

      double[] prob = new double[EvBinary.pow2(parents_indexes.length)];

      EvTriple[] triples = EvBinary.numberOf(population, parents_indexes, n);

      for (int i = 0; i < prob.length; i++) {
        if (triples[i].z == 0) {
          prob[i] = initialProbability;
        } else {
          prob[i] = triples[i].x / (double) triples[i].z;
        }
      }
      node.setProbabilities(prob);
    }
  }

  /**
   * Returns table containing edges on the network in form of Z * 2 int table.
   * 
   * @return Z x 2 table, where Z is number of vertexes. In each row first 
   * value means parental vertex, second one means child vertex.
   */
  public int[][] getEdges() {
    
    int[][] edges = new int[number_of_edges][2];
    
    int it = 0; 
    for (int i = 0; i < vertex_num; i++) {
      for (int j = 0; j < nodes[i].getChildrenCount(); j++) {
        edges[it][1] = nodes[i].getIndex();
        edges[it][0] = (nodes[i].getChildren()[j]).getIndex();
        it++;
      }
    }
    
    return edges;
  }
  

  /**
   * {@inheritDoc}
   */
  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < nodes.length; i++) {
      builder.append(nodes[i].toString());
      builder.append("\n");
    }
    return builder.toString();
  }
}
