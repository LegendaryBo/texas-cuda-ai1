package pl.wroc.uni.ii.evolution.engine.samplealgorithms;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.solutionspaces.EvMessyBinaryVectorSpace;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMGAOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvMessyBinaryVectorObjectiveFunctionWrapper;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;

/**
 * An implementation of the Messy Genetic Algorithm, based on Kalyanmoy Deb and
 * David E. Goldberg's "mGA in C: A Messy Genetic Algorithm in C"
 * 
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 */

public final class EvMGA extends EvAlgorithm<EvMessyBinaryVectorIndividual> {

  /* Size of the problem, length of genotype */
  private int problem_size;

  /* Operator iplementing iteration of Messy Genetic Alghoritm */
  private EvMGAOperator mga_operator;


  /**
   * Constructor of the Messy Genetic Algorithm with full list of parameters.
   * 
   * @param maximum_era - number of eras
   * @param problem_size - vector length
   * @param maximum_population_size - upper limit for generated population size,
   *        if generated cover individuals number exceedes this, an uniformly
   *        random individuals will be selected from them, this option can
   *        preserve from out of memory due to extreme big populations, so it
   *        enables using more eras, NOTE: this option does not belong to the
   *        original mGA
   * @param probability_of_cut - probability of cut an individual, multiplied by
   *        the length of the chromosome, recommended 1.0/(2*problem_size) value
   * @param probability_of_splice - probability of splice two individuals,
   *        recommended high values or 1.0
   * @param probability_of_allelic_mutation - probability of allele negation for
   *        each allele, recommended small or 0.0
   * @param probability_of_genic_mutation - probability of change gene which
   *        allele belongs, recommended small of 0.0, NOTE: this not guarantying
   *        changing gene to a different one, for probability guarantying
   *        changing gene use genic_mutation = changing_genic_mutation *
   *        (problem_length/(problem_length-1)), in the original mGA guarantying
   *        changing gene mutation is used
   * @param thresholding - there will be compared individuals with a number of
   *        common expressed genes larger than expected in random chromosomes
   * @param tie_breaking - shorter individuals have advantage when the objective
   *        function value is the same
   * @param reduced_initial_population - negated template is used for generated
   *        individuals instead all allele combinations.
   * @param keep_era_best_individual - find and keep for the best individual in
   *        whole era time, instead of get it from final era population, NOTE:
   *        this option is an experimental extension, it does not belong to the
   *        original mGA.
   * @param copies - array of numbers of copies generated individuals specified
   *        for all eras
   * @param maximum_generationes - numbers of generations specified for all eras
   * @param juxtapositional_sizes - population sizes in juxtapositional phase
   *        specified for all eras
   */
  public EvMGA(int maximum_era, int problem_size, int maximum_population_size,
      double probability_of_cut, double probability_of_splice,
      double probability_of_allelic_mutation,
      double probability_of_genic_mutation, boolean thresholding,
      boolean tie_breaking, boolean reduced_initial_population,
      boolean keep_era_best_individual, int[] copies,
      int[] maximum_generationes, int[] juxtapositional_sizes) {

    super(1); // Population size is unknown at the moment

    mga_operator =
        new EvMGAOperator(maximum_era, problem_size, maximum_population_size,
            probability_of_cut, probability_of_splice,
            probability_of_allelic_mutation, probability_of_genic_mutation,
            thresholding, tie_breaking, reduced_initial_population,
            keep_era_best_individual, copies, maximum_generationes,
            juxtapositional_sizes);

    // Set iteration number
    for (int i = 0; i < maximum_era; i++)
      iteration_number += maximum_generationes[i];
    this.problem_size = problem_size;
  }


  /**
   * Constructor of the Messy Genetic Algorithm with some parameters
   * 
   * @param maximum_era - number of eras
   * @param problem_size - vector length
   * @param maximum_population_size - upper limit of generated population size,
   *        0 means that there is no limit if generated cover individuals number
   *        exceedes this, an uniformly random individuals will be selected from
   *        them, this option can preserve from out of memory due to extreme big
   *        populations, so it enables using more eras, NOTE: this option does
   *        not belong to the original mGA
   * @param probability_of_cut - probability of cut an individual, multiplied by
   *        the length of the chromosome, recommended 1.0 / (2 * problem_size)
   *        value
   * @param probability_of_splice - probability of splice two individuals,
   *        recommended high values or 1.0
   * @param probability_of_allelic_mutation - probability of allele negation for
   *        each allele, recommended small or 0.0
   * @param probability_of_genic_mutation - probability of change gene which
   *        allele belongs, recommended small of 0.0, NOTE: this not guarantying
   *        changing gene to a different one, for probability guarantying
   *        changing gene use genic_mutation = changing_genic_mutation *
   *        (problem_length/(problem_length-1)), in the original mGA guarantying
   *        changing gene mutation is used
   * @param thresholding - there will be compared individuals with a number of
   *        common expressed genes larger than expected in random chromosomes
   * @param tie_breaking - shorter individuals have advantage when the objective
   *        function value is the same
   * @param reduced_initial_population - negated template is used for generated
   *        individuals instead all allele combinations.
   * @param keep_era_best_individual - find and keep for the best individual in
   *        whole era time, instead of get it from final era population, NOTE:
   *        this option is an experimental extension, it does not belong to the
   *        original mGA.
   * @param maximum_generation - number of generations in every era
   * @param juxtapositional_size - population sizes in juxtapositional phase in
   *        every era
   */
  public EvMGA(int maximum_era, int problem_size, int maximum_population_size,
      double probability_of_cut, double probability_of_splice,
      double probability_of_allelic_mutation,
      double probability_of_genic_mutation, boolean thresholding,
      boolean tie_breaking, boolean reduced_initial_population,
      boolean keep_era_best_individual, int maximum_generation,
      int juxtapositional_size) {

    super(1); // Population size is unknown at the moment

    mga_operator =
        new EvMGAOperator(maximum_era, problem_size, maximum_population_size,
            probability_of_cut, probability_of_splice,
            probability_of_allelic_mutation, probability_of_genic_mutation,
            thresholding, tie_breaking, reduced_initial_population,
            keep_era_best_individual, maximum_generation, juxtapositional_size);

    // Set iteration number
    iteration_number = maximum_era * maximum_generation;
    this.problem_size = problem_size;
  }


  /**
   * Constructor of the Messy Genetic Algorithm with select parameters
   * 
   * @param maximum_era - number of eras
   * @param problem_size - vector length
   * @param copies - array of numbers of copies generated individuals specified
   *        for all eras
   * @param maximum_generationes - number of generations in every era
   * @param juxtapositional_sizes - population sizes in juxtapositional phase in
   *        every era
   */
  public EvMGA(int maximum_era, int problem_size, int[] copies,
      int[] maximum_generationes, int[] juxtapositional_sizes) {

    super(1); // Population size is unknown at the moment

    mga_operator =
        new EvMGAOperator(maximum_era, problem_size, copies,
            maximum_generationes, juxtapositional_sizes);

    // Set iteration number
    for (int i = 0; i < maximum_era; i++)
      iteration_number += maximum_generationes[i];
    this.problem_size = problem_size;
  }


  /**
   * Set solution space. NOTE: If a objective function is set
   * (setWrappedObjectiveFunction()), default solution space is set, so there is
   * not necessity to call this function.
   * 
   * @param solution_space - MessyBinaryVectorSpace
   */
  public void setSolutionSpace(EvMessyBinaryVectorSpace solution_space) {
    this.solution_space = solution_space;
  }


  /**
   * Set wrapper used to evalate objective function for binary vector
   * individuals on messy individuals. NOTE: If a wrapper isn't set, it is taken
   * from the solution space, so there is no necessity (if a wrapper is set in a
   * solution space) to call this function.
   * 
   * @param objective_function_wrapper - wrapper for objective functions
   */
  public void setObjectiveFunction(
      EvMessyBinaryVectorObjectiveFunctionWrapper objective_function_wrapper) {
    this.objective_function = objective_function_wrapper;
  }


  /**
   * Throw Illegal Argument Exception, because the algorithm can work only with
   * wrapped messy objective functions.
   * 
   * @param objective_function - function to evaluate
   */
  @Override
  public void setObjectiveFunction(
      EvObjectiveFunction<EvMessyBinaryVectorIndividual> objective_function) {
    throw new IllegalArgumentException(
        "The algorithm can work only with wrapped messy objective functions");
  }


  /**
   * Sets objective function used to evaluate binary vector individuals. Creates
   * a wrapper and solution space if necessary.
   * 
   * @param objective_function - objective function to evaluate individuals
   */
  public void setWrappedObjectiveFunction(
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function) {

    if (objective_function == null)
      throw new NullPointerException("Objective function is null");

    if (this.objective_function == null)
      this.objective_function =
          new EvMessyBinaryVectorObjectiveFunctionWrapper(objective_function);
    else if (EvMessyBinaryVectorObjectiveFunctionWrapper.class
        .isInstance(this.objective_function))
      ((EvMessyBinaryVectorObjectiveFunctionWrapper) this.objective_function)
          .setObjectiveFunction(objective_function);

    if (solution_space == null)
      solution_space =
          new EvMessyBinaryVectorSpace(this.objective_function, problem_size);
    else
      solution_space.setObjectiveFuntion(this.objective_function);
  }


  /**
   * Initializes algorithm. Set termination condition and checks if solution
   * space is null. Sets an ojective function from solution space if it's not
   * set. Take problem size from solution space and create population.
   */
  @Override
  public void init() {

    // Set termination condition
    if (termination_condition == null)
      setTerminationCondition(new EvMaxIteration<EvMessyBinaryVectorIndividual>(
          iteration_number));

    if (solution_space == null && objective_function == null)
      throw new IllegalStateException(
          "Solution space and objective function are not set!");

    // If solution space is not set, we take it from objective function
    if (solution_space == null)
      solution_space =
          new EvMessyBinaryVectorSpace(objective_function, problem_size);

    // If objective function is not set, we take it from solution space
    if (objective_function == null) {
      if (EvMessyBinaryVectorObjectiveFunctionWrapper.class
          .isInstance(solution_space.getObjectiveFuntion()))
        setObjectiveFunction((EvMessyBinaryVectorObjectiveFunctionWrapper) solution_space
            .getObjectiveFuntion());

      if (objective_function == null)
        throw new IllegalStateException("Objective function is not set!");
    }

    super.addOperatorToBeginning(mga_operator);

    problem_size =
        ((EvMessyBinaryVectorSpace) solution_space).getVectorLength();

    super.init();
    population.setObjectiveFunction(objective_function);

  }

}