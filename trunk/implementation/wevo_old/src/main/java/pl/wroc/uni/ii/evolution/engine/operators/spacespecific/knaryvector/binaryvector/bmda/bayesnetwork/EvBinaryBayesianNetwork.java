package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.bayesnetwork;



import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

/**
 *
 *
 * @author Jarek Fuks (jarek102@gmail.com)
 * @author Zbigniew Nazimek
 */
public final class EvBinaryBayesianNetwork {

  /**
   * Population used in optimalizing nerwork.
   */
  private EvPopulation<EvBinaryVectorIndividual> population;

  /**
   * Vertexes table.
   */
  private EvBinaryBayesianNode[] nodes;

  /**
   * maximum number of parent in single vertex.
   */
  private final int k;

  /**
   *
   */
  private boolean generated_probabilities;


  /**
   * Default constructor.
   *
   * @param max_parents max number of parents in each node
   */
  public EvBinaryBayesianNetwork(final int max_parents) {
    generated_probabilities = false;
    this.k = max_parents;
  }


  /**
   * {@inheritDoc}
   */
  public boolean addEdge(final int from, final int to) {
    if (!legalEdge(from, to)) {
      return false;
    }
    nodes[to].addParent(nodes[from]);
    generated_probabilities = false;
    return true;
  }


  /**
   * {@inheritDoc}
   */
  public boolean removeEdge(final int from, final int to) {
    if (!nodes[to].isParent(nodes[from])) {
      return false;
    }
    nodes[to].removeParent(nodes[from]);
    generated_probabilities = false;
    return true;
  }


  /**
   * {@inheritDoc}
   */
  public EvBinaryVectorIndividual generate() {
    if (!generated_probabilities) {
      generatePropabilities();
      generated_probabilities = true;
    }

    int nodes_size = nodes.length;
    int i = 0;
    for (i = 0; i < nodes_size; i++) {
      nodes[i].initGeneration();
    }

    int[] tab = new int[nodes.length];
    int tab_length = tab.length;
    for (i = 0; i < tab_length; i++) {
      tab[i] = nodes[i].generate();
    }
    EvBinaryVectorIndividual ind = new EvBinaryVectorIndividual(tab);
    ind.setObjectiveFunction(population.get(0).getObjectiveFunction());
    return ind;
  }


  /**
   * generates vector of probabilities.
   */
  private void generatePropabilities() {
    final double initProb = 0.5d;

    int[] parents_indexes;
    int[] all_indexes;
    EvBinaryBayesianNode node;
    for (int n = 0; n < nodes.length; n++) {
      node = nodes[n];
      EvBinaryBayesianNode[] parents = node.getParents();

      parents_indexes = new int[parents.length];
      all_indexes = new int[parents.length + 1];
      for (int j = 0; j < parents_indexes.length; j++) {
        parents_indexes[j] = parents[j].getIndex();
        all_indexes[j] = parents[j].getIndex();
      }
      all_indexes[all_indexes.length - 1] = node.getIndex();

      double[] probabilities = new double[EvBinary.pow2(parents.length)];

      for (int i = 0; i < probabilities.length; i++) {
        int[] parents_value = EvBinary.intToBools(i, parents.length);
        int[] parents_and_true = EvBinary.intToBools(i, parents.length + 1);
        parents_and_true[parents_and_true.length - 1] = 1;

        int count_parents =
            EvBinary.numberOf(population, parents_value, parents_indexes);
        int count_all =
            EvBinary.numberOf(population, parents_and_true, all_indexes);

        if (count_parents == 0) {
          probabilities[i] = initProb;
        } else {
          probabilities[i] = count_all / (double) count_parents;
        }
      }
      node.setProbabilities(probabilities);
    }

  }


  /**
   * {@inheritDoc}
   */
  public int[] getParentsIndexes(final int child_index) {
    EvBinaryBayesianNode[] parents = nodes[child_index].getParents();
    int[] parent_indexes = new int[parents.length];
    int i = 0;
    for (EvBinaryBayesianNode node : parents) {
      parent_indexes[i++] = node.getIndex();
    }
    return parent_indexes;
  }


  /**
   * {@inheritDoc}
   */
  public void initialize(
      final EvPopulation<EvBinaryVectorIndividual> population_) {
    this.population = population_;
    nodes = new EvBinaryBayesianNode[population_.get(0).getDimension()];
    for (int i = 0; i < nodes.length; i++) {
      nodes[i] = new EvBinaryBayesianNode(i);
    }
    generated_probabilities = false;
  }


  /**
   * @param n0 - starting vertex
   * @param n1 - ending vertex
   * @return True, if adding specified edge is legal, false
   * otherwise
   */
  private boolean legalEdge(final int n0, final int n1) {
    return ((!nodes[n1].searchForDescendant(nodes[n0]))
        && (!nodes[n1].isParent(nodes[n0])) && (n0 != n1) && (nodes[n1]
        .getParents().length < k));
  }


  /**
   * {@inheritDoc}
   */
  public int getSize() {
    return nodes.length;
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public EvBinaryBayesianNetwork clone() {
    EvBinaryBayesianNetwork new_net = new EvBinaryBayesianNetwork(k);
    new_net.initialize(population);
    for (EvBinaryBayesianNode node : nodes) {
      for (EvBinaryBayesianNode parent : node.getParents()) {
        new_net.addEdge(parent.getIndex(), node.getIndex());
      }
    }
    return new_net;
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("Current network\n");
    for (int i = 0; i < nodes.length; i++) {
      builder.append(nodes[i]);
    }

    return builder.toString();
  }


  /**
   * {@inheritDoc}
   */
  public int[][] getEdges() {

    //counting number of edges
    int count = 0;
    for (EvBinaryBayesianNode node : nodes) {
      for (EvBinaryBayesianNode parent : node.getParents()) {
        count++;
      }
    }


    // converting into Zx2 table
    int[][] edges = new int[count][2];
    int i = 0;
    for (EvBinaryBayesianNode node : nodes) {
      for (EvBinaryBayesianNode parent : node.getParents()) {
        edges[i][0] = parent.getIndex();
        edges[i][1] = node.getIndex();
        i++;
      }
    }

    return edges;
  }

}
