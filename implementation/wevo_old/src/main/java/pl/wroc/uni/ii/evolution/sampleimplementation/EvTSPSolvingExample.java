package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvBlockSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.EvPermutationInversionMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.EvPermutationPMXCrossover;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.solutionspaces.EvPermutationSpace;

/**
 * Class implementing an example of solving a TSP problem. More detailed
 * information can be found in corresponding tutorial on main wEvo website.
 * 
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 * @author Karol "Asgaroth" Stosiek (karol.stosiek@gmail.com)
 */

public class EvTSPSolvingExample {
  public static void main(String[] args) {

    /*
     * an instance of a symmetric TSP problem, encoded by a matrix of distances.
     */
    double distances[][] =
        new double[][] { {0, 20, 35, 24}, {20, 0, 39, 30}, {35, 39, 0, 19},
            {24, 30, 19, 0}};

    // creating new evolutionary task
    EvAlgorithm<EvPermutationIndividual> evolutionary_algorithm =
        new EvAlgorithm<EvPermutationIndividual>(100);

    // creating the solution space
    EvPermutationSpace solution_space = new EvPermutationSpace(4);

    /*
     * creating an objective function associated with distances matrix
     */
    EvTSPSolvingObjectiveFunction objective_function =
        new EvTSPSolvingObjectiveFunction(distances);

    solution_space.setObjectiveFuntion(objective_function);

    /*
     * setting solution space and termination condition
     */
    evolutionary_algorithm.setSolutionSpace(solution_space);
    evolutionary_algorithm
        .setTerminationCondition(new EvMaxIteration<EvPermutationIndividual>(
            100));

    /* adding operators to algorithm main loop */
    evolutionary_algorithm
        .addOperator(new EvBlockSelection<EvPermutationIndividual>(30));
    evolutionary_algorithm.addOperator(new EvPermutationPMXCrossover(1, 2));
    evolutionary_algorithm
        .addOperator(new EvPermutationInversionMutation(0.02));

    // initialize and run
    evolutionary_algorithm.init();
    evolutionary_algorithm.run();

    // get best individual
    EvPermutationIndividual best =
        (EvPermutationIndividual) evolutionary_algorithm.getBestResult();

    System.out.println(best);
  }

}
