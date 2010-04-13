package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import java.util.ArrayList;
import java.util.Iterator;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.bayesian.EvBayesianNetworkStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga.EvPopulationGenerating;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga.EvStructureDiscover;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvBayesianOperator;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Class implementing ECGA algorithm with {@link EvTournamentSelection} as
 * operator. <br>
 * This operator uses three operators: EvTournamenSelection,
 * EvStructureDiscover, EvPopulationGenerating. <br>
 * At first selection is done by EvTournamentSelection. Then EvStructureDiscover
 * using population discover the best partition of chromosome into blocks. Then
 * EvPopulationGeneration using this partition randomly shuffles blocks between
 * individuals.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * Krzysztof Sroka (krzysztof.sroka@gmail.com)
 * Marek Chrusciel (bse@gmail.com)
 * 
 * PROFILED
 *         09.IV.2007
 */
public class EvBinaryVectorECGAOperator implements
    EvOperator<EvBinaryVectorIndividual>, EvBayesianOperator {

  /**
   * How many individuals are competing each other in single instance
   * tournament. Only one individual is winning.
   */
  private int tournament_size;

  /**
   * true - operator will try to improve building block used in previous 
   * iteration.
   * false - operator will try to discover building block from the beggining
   * during each iteration.
   */
  private boolean use_previous_structure;

  /**
   * Solution space used by the operator.
   */ 
  private EvBinaryVectorSpace solution_space;
  
  /**
   * statistic storage object in which we store information about building 
   * blocks during every iteration.
   */
  private EvPersistentStatisticStorage storage = null;

  /**
   * Object storing building blocks in current iteration.
   */
  private EvStructureDiscover struct_update = null;
  
  /**
   * we use EvStructureDiscover, EvPopulationGenerating and
   * EvTournamentSelection in this algorithm during every iterations. Those
   * operators are stored here:
   */
  private ArrayList<EvOperator<EvBinaryVectorIndividual>> operators =
      new ArrayList<EvOperator<EvBinaryVectorIndividual>>();


  /**
   * @param tournament_size_ - how many individuals compete in each instance 
   * of the tournament. Only one individual wins the tournament.
   * @param use_prev_structure
   *        <ul>
   *        <li> if <code> true </code> then discovery of the best structure
   *        uses structure from previous iteration
   *        <li> if <code> false </code> then discovery of the best structure is
   *        done from initial structure.<BR>
   *        </ul>
   * @param solution_space_ - solution space used by operator
   */
  public EvBinaryVectorECGAOperator(final int tournament_size_,
      final boolean use_prev_structure, 
      final EvBinaryVectorSpace solution_space_) {

    /*
     * check if arguments are ok.
     */
    if (tournament_size_ <= 0) {
      throw new IllegalArgumentException(
          "ECGA accepts only tournament_size as a parameter"
              + " which must be a positive Integer");
    }
    if (solution_space_ == null) {
      throw new IllegalArgumentException("solution_space cannot be null");
    }

    this.solution_space = solution_space_;
    this.tournament_size = tournament_size_;
    this.use_previous_structure = use_prev_structure;
    // initializing operators used by this operator
    init();
  }


  /**
   * {@inheritDoc}
   */
  public EvPopulation<EvBinaryVectorIndividual> apply(
      final EvPopulation<EvBinaryVectorIndividual> population) {

    EvPopulation<EvBinaryVectorIndividual> pop = population;
    
    /*
     * error checking
     */
    if (population == null) {
      throw new IllegalArgumentException(" Applied population cannot be null ");
    }
    if (population.size() == 0) {
      throw new IllegalArgumentException(
          " Applied population must contain at leat one individual ");
    }

    /*
     * we apply 3 operators: EvStructureDiscover, EvPopulationGenerating and
     * EvTournamentSelection created duning initialization to given population
     */
    for (Iterator<EvOperator<EvBinaryVectorIndividual>> iterator =
        operators.iterator(); iterator.hasNext();) {
      EvOperator<EvBinaryVectorIndividual> operator = iterator.next();
      pop =
          (EvPopulation<EvBinaryVectorIndividual>) operator.apply(pop);
      /*
       * Some very basic sanity check. Population cannot be null or empty.
       */
      if (population == null) {
        throw new IllegalStateException("Operator " + operator.getClass()
            + " has returned null instead of valid population");
      }

      if (population.size() == 0) {
        throw new IllegalStateException("Operator " + operator.getClass()
            + " has shrunk population to zero.");
      }
    }
    
    if (storage != null) {
      int[][] edges = struct_update.getEdges();
      
      EvBayesianNetworkStatistic stat = new EvBayesianNetworkStatistic(edges, 
          population.get(0).getDimension());
      storage.saveStatistic(stat);
    }

    return pop;

  }


  /**
   * Initializing operators used by this operator.
   */ 
  private void init() {

    struct_update =
        new EvStructureDiscover((EvBinaryVectorSpace) solution_space,
            use_previous_structure);
    EvPopulationGenerating pop_gen =
        new EvPopulationGenerating(struct_update,
            (EvBinaryVectorSpace) solution_space);

    operators.add((EvOperator<EvBinaryVectorIndividual>) 
        (new EvTournamentSelection<EvBinaryVectorIndividual>(
            tournament_size, 1)));
    operators.add(struct_update);
    operators.add(pop_gen);

  }


  /**
   * {@inheritDoc}
   */
  public void collectBayesianStats(final EvPersistentStatisticStorage stor) {
    storage = stor;
    
  }

}
