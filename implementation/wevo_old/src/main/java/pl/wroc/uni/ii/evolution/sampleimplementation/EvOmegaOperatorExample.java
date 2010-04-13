package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.omega.EvOmegaOperator;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.solutionspaces.EvOmegaSpace;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.omega.EvOmegaSalesman;

/**
 * Solution of TSP by omega operator.
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaOperatorExample {
  public static void main(String[] args) {
    /*
     * an instance of a symmetric TSP problem, encoded by a matrix of distances.
     */
    double distances[][] =
        new double[][] { {0, 20, 35, 24}, {20, 0, 39, 30}, {35, 39, 0, 19},
            {24, 30, 19, 0}};

    // size of initial population
    final int population_size = 100;

    // constant individuals length reduction ratio
    final double reduction_ratio = 0.5;

    // cut probility for cut operator
    final double cut_probability = 0.05;

    // splice probability for splice operator
    final double splice_probability = 0.95;

    // locus mutation factor
    final double locus_mutation = 0.002;

    // allelic mutation factor
    final double allelic_mutatation = 0.002;

    // maximum number of eras
    final int era_max = 4;

    // size of the problem -- genotype length
    final int problem_size = 4;

    // maximum number of juxtapositional loops
    final int juxtapositional_loop_max = 3;

    // growing factor
    double growing_population_size_factor = 2.0;

    // maximum number of epochs
    final int epoch_max = 2;

    // an objective function for TSP
    EvOmegaSalesman salesman_obj = new EvOmegaSalesman(distances);

    // creating an algorithm instance
    EvAlgorithm<EvOmegaIndividual> evolutionary_algorithm =
        new EvAlgorithm<EvOmegaIndividual>(population_size);

    // creating solution space for the algorithm
    EvOmegaSpace solution_space = new EvOmegaSpace(problem_size, salesman_obj);

    // setting the solution space
    evolutionary_algorithm.setSolutionSpace(solution_space);

    // setting termination condition
    evolutionary_algorithm
        .setTerminationCondition(new EvMaxIteration<EvOmegaIndividual>(
            epoch_max));

    // adding Omega operator to algorithm
    evolutionary_algorithm.addOperator(new EvOmegaOperator(population_size,
        problem_size, reduction_ratio, era_max, juxtapositional_loop_max,
        growing_population_size_factor, cut_probability, splice_probability,
        locus_mutation, allelic_mutatation));

    // algorithm initialisation
    evolutionary_algorithm.init();

    // starting the algorithm
    evolutionary_algorithm.run();

    // choosing the best individual
    EvOmegaIndividual best =
        (EvOmegaIndividual) evolutionary_algorithm.getBestResult();

    // printing out our results
    System.out.println(best + "\n" + best.getFenotype() + "\n"
        + -best.getObjectiveFunctionValue());
  }
}