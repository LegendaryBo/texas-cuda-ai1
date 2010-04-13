package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorUMDAOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorSpace;

/**
 * Example of UMDAc algorithm on OneMax function, using K-Best selection. This
 * example shows an exemplary usage of UMDAc algorithm (<code>EvRealVectorUMDAOperator</code>),
 * using K-Best selection and OneMax objective function. Prints out the best
 * individual found to <code>System.out</code>.
 * 
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public final class EvRealUMDAExample {

  /**
   * Protects from creating an instance of this utility class.
   */
  private EvRealUMDAExample() {
  }


  /**
   * Runs the UMDAc algorithm.
   * 
   * @param args ignored
   */
  public static void main(final String[] args) {
    // off MagicNumber
    // parameters to the example
    int population_size = 100;
    int individual_dimension = 10;
    int max_iteration_count = 50;
    int k_best = 20;
    // on MagicNumber

    EvAlgorithm<EvRealVectorIndividual> algorithm =
        new EvAlgorithm<EvRealVectorIndividual>(population_size);

    EvObjectiveFunction<EvRealVectorIndividual> objective_fn =
        new EvRealOneMax<EvRealVectorIndividual>();

    EvSolutionSpace<EvRealVectorIndividual> sol_space =
        new EvRealVectorSpace(objective_fn, individual_dimension);

    algorithm.setObjectiveFunction(objective_fn);
    algorithm.setSolutionSpace(sol_space);
    algorithm
        .setTerminationCondition(new EvMaxIteration<EvRealVectorIndividual>(
            max_iteration_count));

    algorithm.addOperator(new EvKBestSelection<EvRealVectorIndividual>(k_best));
    algorithm.addOperator(new EvRealVectorUMDAOperator(population_size));

    algorithm.init();
    algorithm.run();

    EvIndividual best = algorithm.getBestResult();
    System.out.println("Best individual: " + best);
  }
}
