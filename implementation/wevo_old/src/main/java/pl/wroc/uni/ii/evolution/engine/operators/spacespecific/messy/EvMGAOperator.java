package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.solutionspaces.EvMessyBinaryVectorSpace;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvPrimordialOperator;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvJuxtapositionalOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvMessyBinaryVectorObjectiveFunctionWrapper;

/**
 * An implementation of the Messy Genetic Algorithm, based on Kalyanmoy Deb and
 * David E. Goldberg's "mGA in C: A Messy Genetic Algorithm in C" This operator
 * works only with EvMessyBinaryVectorObjectiveFunctionWrapper and it use own
 * EvMessyBinaryVectorSpace to generate individuals.
 * 
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 */

public final class EvMGAOperator implements
    EvOperator<EvMessyBinaryVectorIndividual> {

  /* Upper limit of the era */
  private int maximum_era = 3;

  /* Size of the problem, length of genotype */
  private int problem_size = 50;

  /* Upper limit of the population size, 0 means that there is no limit */
  private int maximum_population_size = 0;

  /* If reduced population is desired */
  private boolean reduced_initial_population = true;

  /*
   * Search for best individual in whole era, instead of in final era population
   */
  private boolean keep_era_best_individual = false;

  /*
   * This array contains the number of duplicates of each individual in the
   * initial population specified for each era
   */
  private int[] copies = new int[] {5, 1, 1};

  /* Upper limit of the generationes specified for each era */
  private int[] maximum_generationes = new int[] {30, 50, 100};

  /* Current era */
  private int era;

  /* Current generation */
  private int generation;

  /* Template is used to fill up the underspecified genes in a chromosome */
  private boolean[] template;

  /* Objective function value of the template */
  private double templatefitness;

  /* The duration of the primordial phase */
  private int primordial_generation;

  /* The best found individual of current era */
  private EvMessyBinaryVectorIndividual best_individual;

  /* Primordial phase */
  private EvPrimordialOperator<EvMessyBinaryVectorIndividual> primordial;

  /* Juxtapositional phase */
  private EvJuxtapositionalOperator juxtapositional;

  /* Solution space */
  private EvMessyBinaryVectorSpace solution_space;

  /* Wrapper for evaluating objective functions for binary individual */
  private EvMessyBinaryVectorObjectiveFunctionWrapper objective_function_wrapper;

  private EvPopulation<EvMessyBinaryVectorIndividual> population;


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
  public EvMGAOperator(int maximum_era, int problem_size,
      int maximum_population_size, double probability_of_cut,
      double probability_of_splice, double probability_of_allelic_mutation,
      double probability_of_genic_mutation, boolean thresholding,
      boolean tie_breaking, boolean reduced_initial_population,
      boolean keep_era_best_individual, int[] copies,
      int[] maximum_generationes, int[] juxtapositional_sizes) {

    if (maximum_era < 1)
      throw new IllegalArgumentException(
          "Maximum era must be positive integer.");

    initialization(maximum_era, problem_size, maximum_population_size,
        probability_of_cut, probability_of_splice,
        probability_of_allelic_mutation, probability_of_genic_mutation,
        thresholding, tie_breaking, reduced_initial_population,
        keep_era_best_individual, copies, maximum_generationes,
        juxtapositional_sizes);
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
   * @param maximum_generation - number of generations in every era
   * @param juxtapositional_size - population sizes in juxtapositional phase in
   *        every era
   */
  public EvMGAOperator(int maximum_era, int problem_size,
      int maximum_population_size, double probability_of_cut,
      double probability_of_splice, double probability_of_allelic_mutation,
      double probability_of_genic_mutation, boolean thresholding,
      boolean tie_breaking, boolean reduced_initial_population,
      boolean keep_era_best_individual, int maximum_generation,
      int juxtapositional_size) {

    if (maximum_era < 1)
      throw new IllegalArgumentException(
          "Maximum era must be positive integer.");

    int[] copies = new int[maximum_era];
    int[] maximum_generationes = new int[maximum_era];
    int[] juxtapositional_sizes = new int[maximum_era];

    for (int i = 0; i < maximum_era; i++) {
      copies[i] = 1;
      maximum_generationes[i] = maximum_generation;
      juxtapositional_sizes[i] = juxtapositional_size;
    }

    initialization(maximum_era, problem_size, maximum_population_size,
        probability_of_cut, probability_of_splice,
        probability_of_allelic_mutation, probability_of_genic_mutation,
        thresholding, tie_breaking, reduced_initial_population,
        keep_era_best_individual, copies, maximum_generationes,
        juxtapositional_sizes);
  }


  /**
   * Constructor of the Messy Genetic Algorithm with select parameters.
   * 
   * @param maximum_era - number of eras
   * @param problem_size - vector length
   * @param copies - array of numbers of copies generated individuals specified
   *        for all eras
   * @param maximum_generationes - number of generations in every era
   * @param juxtapositional_sizes - population sizes in juxtapositional phase in
   *        every era
   */
  public EvMGAOperator(int maximum_era, int problem_size, int[] copies,
      int[] maximum_generationes, int[] juxtapositional_sizes) {

    initialization(maximum_era, problem_size, maximum_population_size,
        1.0 / (2 * problem_size), 1.0, 0.001, 0.0, false, true,
        reduced_initial_population, keep_era_best_individual, copies,
        maximum_generationes, juxtapositional_sizes);
  }


  /*
   * Checks and sets parameters in the algorithm and create primordial and
   * juxtapositional operators.
   */
  private void initialization(int maximum_era, int problem_size,
      int maximum_population_size, double probability_of_cut,
      double probability_of_splice, double probability_of_allelic_mutation,
      double probability_of_genic_mutation, boolean thresholding,
      boolean tie_breaking, boolean reduced_initial_population,
      boolean keep_era_best_individual, int[] copies,
      int[] maximum_generationes, int[] juxtapositional_sizes) {

    if (copies.length != maximum_era)
      throw new IllegalArgumentException(
          "Length of copies and maximum era must be equal");

    if (maximum_generationes.length != maximum_era)
      throw new IllegalArgumentException(
          "Length of maximum generationes and maximum era must be equal");

    if (juxtapositional_sizes.length != maximum_era)
      throw new IllegalArgumentException(
          "Length of juxtapositional_sizes and maximum era must be equal");

    if (maximum_population_size != 0)
      for (int i = 0; i < maximum_era; i++)
        if (juxtapositional_sizes[i] > maximum_population_size)
          throw new IllegalArgumentException(
              "Juxtapositional size can't be greater then "
                  + "maximum population size");

    this.maximum_era = maximum_era;
    this.problem_size = problem_size;
    this.maximum_population_size = maximum_population_size;
    this.reduced_initial_population = reduced_initial_population;
    this.keep_era_best_individual = keep_era_best_individual;
    this.copies = copies;
    this.maximum_generationes = maximum_generationes;
    era = 0;

    primordial =
        new EvPrimordialOperator<EvMessyBinaryVectorIndividual>(problem_size,
            thresholding, tie_breaking, copies, juxtapositional_sizes);

    juxtapositional =
        new EvJuxtapositionalOperator(probability_of_cut,
            probability_of_splice, probability_of_allelic_mutation,
            probability_of_genic_mutation, thresholding, tie_breaking);
  }


  /**
   * A single iteration of MessyGA algorithm.
   */
  public EvPopulation<EvMessyBinaryVectorIndividual> apply(
      EvPopulation<EvMessyBinaryVectorIndividual> population) {

    this.population = population;

    if (era == 0)
      init();
    else if (era < maximum_era)
      if (generation > maximum_generationes[era - 1]) {
        if (!keep_era_best_individual)
          best_individual = this.population.getBestResult();
        assignBestIndividualToTemplate();
        era++;
        if (era > maximum_era)
          era = 1;
        initEra();
        best_individual = null;
      }

    if (generation <= primordial_generation)
      this.population = primordial.apply(this.population);
    else
      this.population = juxtapositional.apply(this.population);

    if (keep_era_best_individual) {
      EvMessyBinaryVectorIndividual best_individual_of_population =
          this.population.getBestResult();

      if (best_individual == null
          || best_individual.getObjectiveFunctionValue() < best_individual_of_population
              .getObjectiveFunctionValue())
        best_individual = best_individual_of_population;
    }

    generation++;

    return this.population;
  }


  /*
   * Initializes the algorithm. Gets an objective function, creates
   * MessyBinaryVectorSpace and creates random template and initializes first
   * era.
   */
  private void init() {

    if (EvMessyBinaryVectorObjectiveFunctionWrapper.class.isInstance(population
        .get(0).getObjectiveFunction()))
      objective_function_wrapper =
          (EvMessyBinaryVectorObjectiveFunctionWrapper) population
              .getBestResult().getObjectiveFunction();
    else
      throw new IllegalArgumentException("" + "MGAOperator works only with "
          + "EvMessyBinaryVectorObjectiveFunctionWrapper");

    solution_space =
        new EvMessyBinaryVectorSpace(objective_function_wrapper, problem_size);

    template =
        EvMessyBinaryVectorObjectiveFunctionWrapper
            .getRandomTemplate(problem_size);
    setTemplateFitness();

    era = 1;
    initEra();
    best_individual = null;
  }


  /*
   * Initializes era. Generates initial population and updates variable
   * primordial_generation.
   */
  private void initEra() {
    objective_function_wrapper.setTemplate(template);

    // Generate initial population
    initPopulation();

    primordial_generation = primordial.setupNewEra(population, templatefitness);

    generation = 1;
  }


  /*
   * Generates initial population
   */
  private void initPopulation() {
    if (reduced_initial_population)
      population =
          new EvPopulation<EvMessyBinaryVectorIndividual>(solution_space
              .generateCoverIndividuals(era, copies[era - 1],
                  maximum_population_size, getNegatedTemplate()));
    else
      population =
          new EvPopulation<EvMessyBinaryVectorIndividual>(solution_space
              .generateCoverIndividuals(era, copies[era - 1],
                  maximum_population_size));

    population.setObjectiveFunction(objective_function_wrapper);
  }


  /*
   * Sets value of objective function for template
   */
  private void setTemplateFitness() {
    // Create messy individual to evaluate templatefitness
    ArrayList<Integer> genes = new ArrayList<Integer>(problem_size);
    ArrayList<Boolean> alleles = new ArrayList<Boolean>(problem_size);
    for (int i = 0; i < problem_size; i++) {
      genes.add(i, i);
      alleles.add(i, template[i]);
    }
    EvMessyBinaryVectorIndividual individual =
        new EvMessyBinaryVectorIndividual(problem_size, genes, alleles);
    templatefitness = objective_function_wrapper.evaluate(individual);
  }


  /*
   * Negates values of template. @return array with negated values of template
   */
  private boolean[] getNegatedTemplate() {
    boolean[] negated_template = new boolean[problem_size];
    for (int i = 0; i < problem_size; i++)
      negated_template[i] = !template[i];
    return negated_template;
  }


  /*
   * Assigns the best individual to template.
   */
  private void assignBestIndividualToTemplate() {
    templatefitness = objective_function_wrapper.evaluate(best_individual);
    ArrayList<Boolean>[] genotype = best_individual.getGenotype();
    for (int i = 0; i < problem_size; i++)
      if (genotype[i].size() > 0)
        template[i] = genotype[i].get(0);
  }

}