package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.bayesian.EvBayesianNetworkStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.comitstructs.EvGraph;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.comitstructs.EvTree;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvBayesianOperator;

/**
 * Class implementing Comit operator for population of BinaryIndividual.<BR>
 * <BR>
 * Operator builds new population from probability tree made of Population given
 * in apply method in following steps:<BR> - create matrix of similarity of
 * individuals<BR> - create graph from matrix with weights containig values
 * from matrix<BR> - create maximum spinning tree from graph<BR> - randomly
 * select node in tree spinning tree<BR> - create probability tree with root as
 * selected node and add probability values in every node comparing individuals
 * representing connected nodes.<BR>
 * 
 * @author Kacper Gorski (admin@34all.org)
 * @author Olgierd Humenczuk
 */
public class EvBinaryVectorCOMITOperator implements
    EvOperator<EvBinaryVectorIndividual>, EvBayesianOperator {

  /**
   * size of new population.
   */
  private int new_population_size = 0;
  
  /**
   * Object storing graph history.
   */
  private EvPersistentStatisticStorage storage;


  /**
   * Constructor that can set size of the new population.
   * 
   * @param new_population_size_ - size of population returned by 
   * the oeprator
   */
  public EvBinaryVectorCOMITOperator(final int new_population_size_) {
    this.new_population_size = new_population_size_;
  }


  /**
   * Run this operator on given population and return a new - better one.
   * 
   * @param population - input population.
   * @return new better population generated from the tree of probability
   *         created from given population, new population size is given as
   *         constructor param.
   */
  @SuppressWarnings("unchecked")
  public EvPopulation<EvBinaryVectorIndividual> apply(
      final EvPopulation<EvBinaryVectorIndividual> population) {

    if (new_population_size <= 0) {
      throw new IllegalArgumentException(
          "Size of new population should be greater then zero");
    }

    int ind_size = population.get(0).getDimension();

    // creating matrix of similarity from individual characteristics
    int[][] matrix = new int[ind_size][ind_size];
    for (int i = 0; i < ind_size; i++) {
      for (int j = 0; j < i; j++) {
        matrix[i][j] = getNoSameBits(population, i, j);
      }
    }

    // creating graph from matrix
    EvGraph graph = new EvGraph(matrix);

    // creating maximum spanning tree from graph
    EvTree tree = graph.getMaximumSpanningTree();
    
    // building tree of probability from population
    tree.setProbability(population);

    // generating new population
    EvBinaryVectorIndividual[] ind_table =
        new EvBinaryVectorIndividual[new_population_size];
    for (int i = 0; i < new_population_size; i++) {
      ind_table[i] = new EvBinaryVectorIndividual(ind_size);
    }
    tree.randomizeIndividual(ind_table);
    EvPopulation<EvBinaryVectorIndividual> new_population =
        new EvPopulation<EvBinaryVectorIndividual>(ind_table);

    new_population.setObjectiveFunction(population.get(0)
        .getObjectiveFunction());
    
    
    
    // storing network statistics for futher investigation
    if (storage != null) {
      ArrayList<EvTree> vertexes = new ArrayList<EvTree>();
      vertexes.add(tree);
      
      int[][] edges = new int[ind_size][2];
      
      int i = 0;
      while (vertexes.size() != 0) {
        EvTree vertex = vertexes.get(0);
        for (EvTree child : vertex.getChildren()) {
          edges[i][0] = vertex.getLabel();
          edges[i][1] = child.getLabel();
          vertexes.add(child);
          i++;
        }
        vertexes.remove(vertex);
      }
      
      EvBayesianNetworkStatistic stat = 
        new EvBayesianNetworkStatistic(edges, ind_size);
      
      storage.saveStatistic(stat);
    }
    
    
    
    return new_population;
  }


  /**
   * Little help function that count number of same bits at specified location
   * in population matrix.
   * 
   * @param population - population object
   * @param compare_bit - the row to be compared
   * @param compare_to_bit - the row to be compared to
   * @return number of same bits in that two rows
   */
  private int getNoSameBits(
      final EvPopulation<EvBinaryVectorIndividual> population,
      final int compare_bit, final int compare_to_bit) {
    int col_size = population.size();
    int same_bits = 0;

    EvBinaryVectorIndividual[] individual_table =
        population.toArray(new EvBinaryVectorIndividual[population.size()]);

    // for every column in matrix get desired bit in desire row and compare it
    // with
    // another bit in another row in the same individual
    for (int it = 0; it < col_size; it++) {
      if (individual_table[it].getGene(compare_bit) == individual_table[it]
          .getGene(compare_to_bit)) {
        same_bits++;
      }
    }

    return same_bits;
  }


  /**
   * {@inheritDoc}
   */
  public void collectBayesianStats(
      final EvPersistentStatisticStorage storage_) {
    storage = storage_;
  }

}
