package pl.wroc.uni.ii.evolution.objectivefunctions.messy;

import java.util.ListIterator;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Wrapper class for evaluating objective functions for BinaryVectorIndividual
 * on MessyBinaryVectorIndividual class instances. For underspecified genes it
 * uses the template to fill up them, or if the template is not set, it gets
 * uniformly random alleles for them. For overspecified genes it uses specified
 * number of randomly generated combination of alleles and uses the best, or if
 * checks number is 0, it gets first allele of each gene in a chromosome.
 * 
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 */

public class EvMessyBinaryVectorObjectiveFunctionWrapper implements
    EvObjectiveFunction<EvMessyBinaryVectorIndividual> {

  private static final long serialVersionUID = -3558021101397905305L;

  // Template used to fill up the underspecifed genes in a chromosome
  private boolean[] template;

  // Number of random derivaties of MessyBinaryVectorIndividual to check
  private int checks_number = 0;

  // Function to evaluate
  private EvObjectiveFunction<EvBinaryVectorIndividual> function;


  /**
   * Constructor creates wrapper with specified objective function, template and
   * checks number.
   * 
   * @param function - objective function to evaluate
   * @param template - the template to fill up the underspecifed genes in a
   *        chromosome
   * @param checks_number - number of checks
   */
  public EvMessyBinaryVectorObjectiveFunctionWrapper(
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function,
      boolean[] template, int checks_number) {

    setObjectiveFunction(objective_function);
    setTemplate(template);
    setChecksNumber(checks_number);
  }


  /**
   * Constructor creates wrapper with specified objective function and template,
   * the checks number is set to 0 so the first allele of each gene will be
   * used.
   * 
   * @param function - objective function to evaluate
   * @param template - the template to fill up the underspecifed genes in a
   *        chromosome.
   */
  public EvMessyBinaryVectorObjectiveFunctionWrapper(
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function,
      boolean[] template) {

    setObjectiveFunction(objective_function);
    ;
    setTemplate(template);
  }


  /**
   * Constructor creates wrapper with specified objective function and checks
   * number, the template is not set, so the random alleles will be used
   * instead.
   * 
   * @param function - objective function to evaluate
   * @param checks_number - number of checks
   */
  public EvMessyBinaryVectorObjectiveFunctionWrapper(
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function,
      int checks_number) {

    setObjectiveFunction(objective_function);
    setChecksNumber(checks_number);
  }


  /**
   * Constructor creates wrapper with objective function, the template is not
   * set and checks number is set to 0, so there will be used random alleles for
   * underspecified genes, and first allele of overspecified genes.
   * 
   * @param objective_function - objective function to evaluate
   */
  public EvMessyBinaryVectorObjectiveFunctionWrapper(
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function) {

    setObjectiveFunction(objective_function);
  }


  /**
   * Gets new uniformly random template. It is a binary vector with randomly
   * choosen values.
   * 
   * @param length - length of the template
   * @return random template
   */
  static public boolean[] getRandomTemplate(int length) {
    boolean[] template = new boolean[length];
    for (int i = 0; i < length; i++)
      template[i] = EvRandomizer.INSTANCE.nextBoolean();
    return template;
  }


  /**
   * Sets the objective function.
   * 
   * @param objective_function - the objective function
   */
  public void setObjectiveFunction(
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function) {
    if (objective_function == null)
      throw new NullPointerException("Objective function is null");
    this.function = objective_function;
  }


  /**
   * Gets the objective function.
   * 
   * @return the objective function
   */
  public EvObjectiveFunction<EvBinaryVectorIndividual> getObjectiveFunction() {
    return function;
  }


  /**
   * Sets the template. The template is used to fill up the underspecifed genes
   * in a chromosome. If it is null then random alleles will be used instead.
   * 
   * @param template - the template
   */
  public void setTemplate(boolean[] template) {
    this.template = template;
  }


  /**
   * Gets the template or returns null if not set.
   * 
   * @return template
   */
  public boolean[] getTemplate() {
    return template;
  }


  /**
   * Sets the number of checked random derived allele combinations. If it is 0
   * then first allele of overspecified genes will be used.
   * 
   * @param checks_number - number of checks
   */
  public void setChecksNumber(int checks_number) {
    if (checks_number < 0)
      throw new IllegalArgumentException(
          "checks_number must be a natural number");
    this.checks_number = checks_number;
  }


  /**
   * Gets the number of checks.
   * 
   * @return number of checks
   */
  public int getChecksNumber() {
    return checks_number;
  }


  /**
   * Gets phenotype of given individual, the phenotype is set of genes values
   * which are used to calculate objective function value.
   * 
   * @param individual - messy individual to get phenotype
   * @return phenotype
   */
  public boolean[] getPhenotype(EvMessyBinaryVectorIndividual individual) {

    int[] int_phenotype;

    if (checks_number == 0)
      // Return phenotype of first binary individual derived from individual
      int_phenotype = getFirstDerived(individual).getGenes();
    else
      // Return phenotype of the best random derived from individual
      int_phenotype = getBestDerived(individual).getGenes();

    boolean[] phenotype = new boolean[int_phenotype.length];
    for (int i = 0; i < int_phenotype.length; i++)
      phenotype[i] = (int_phenotype[i] > 0) ? true : false;

    return phenotype;
  }


  /**
   * Evaluates objective function value of given individual. This calulates the
   * phenotype and uses it as parameter for evaluating objective function value.
   * 
   * @param individual - MessyIndividual to evaulate.
   * @return objective function value for given individual
   */
  public double evaluate(EvMessyBinaryVectorIndividual individual) {

    if (template != null && template.length != individual.getGenotypeLength())
      throw new IllegalArgumentException(
          "Individual length and template length must be equal");

    // Return value of binary individual derived from individual
    if (checks_number == 0)
      return function.evaluate(getFirstDerived(individual));

    // Return the best value of checks_number random derived individuals
    return function.evaluate(getBestDerived(individual));
  }


  /*
   * Check checks_number random derived individuals and return the best. @param
   * individual - messy individual to derive @return the best binary vector
   * individual
   */
  private EvBinaryVectorIndividual getBestDerived(
      EvMessyBinaryVectorIndividual individual) {

    int genotype_length = individual.getGenotypeLength();
    int[] alleles_count = new int[genotype_length];
    ListIterator<Integer> genes_iterator = individual.getGenes().listIterator();

    // Count number of alleles for every gene
    for (int i = 0; i < genotype_length; i++)
      alleles_count[i] = 0;
    while (genes_iterator.hasNext()) {
      int gene = genes_iterator.next();
      alleles_count[gene]++;
    }

    // Return the best of checks_number random derived individual
    double best_value = Double.NEGATIVE_INFINITY;
    EvBinaryVectorIndividual best_ind =
        new EvBinaryVectorIndividual(genotype_length);

    for (int i = 0; i < checks_number; i++) {
      EvBinaryVectorIndividual random_ind =
          getRandomDerived(individual, alleles_count);

      double random_ind_value = function.evaluate(random_ind);
      if (random_ind_value > best_value) {
        best_ind = random_ind;
        best_value = random_ind_value;
      }
    }

    return best_ind;
  }


  /*
   * Gets BinaryVectorIndividual with randomly choosen alleles for
   * underspecified genes from given MessyBinaryVectorIndividual. @param
   * individual - individual to derive @param alleles_count - number of alleles
   * in the genotype of each gene @return derived individual
   */
  private EvBinaryVectorIndividual getRandomDerived(
      EvMessyBinaryVectorIndividual individual, int[] alleles_count) {

    int genotype_length = alleles_count.length;

    // Gets random alleles of individual
    Boolean[] genes = getRandomAlleles(individual, alleles_count);

    // Gets underspecified genes
    getUnderspecifiedGenes(genes);

    // Create and return BinaryVectorIndividual
    int[] new_genes = new int[genotype_length];
    for (int i = 0; i < genotype_length; i++)
      new_genes[i] = genes[i] ? 1 : 0;

    EvBinaryVectorIndividual new_individual =
        new EvBinaryVectorIndividual(new_genes);
    new_individual.setObjectiveFunction(function);
    return new_individual;
  }


  /*
   * Gets BinaryVectorIndividual with choosen first allele of overspecified
   * genes from given MessyIndividual. The underspecified genes values are get
   * from the template or they are choosen randomly if the template is not set.
   * @param individual - individual to derive @return derived individual
   */
  private EvBinaryVectorIndividual getFirstDerived(
      EvMessyBinaryVectorIndividual individual) {

    // Gets first values of expressed genes
    Boolean[] genes = getFirstAlleles(individual);

    // Gets underspecified genes
    getUnderspecifiedGenes(genes);

    // Create and return binary vector individual
    int genotype_length = individual.getGenotypeLength();
    int[] new_genes = new int[genotype_length];
    for (int i = 0; i < genotype_length; i++)
      new_genes[i] = genes[i] ? 1 : 0;

    EvBinaryVectorIndividual new_individual =
        new EvBinaryVectorIndividual(new_genes);
    new_individual.setObjectiveFunction(function);
    return new_individual;
  }


  /*
   * Gets randomly selected allele of each gene. @param individual - messy
   * individual to get randomly selected alleles @param alleles_count - number
   * of alleles in the genotype of each gene @return array of random alleles
   */
  private Boolean[] getRandomAlleles(EvMessyBinaryVectorIndividual individual,
      int[] alleles_count) {

    int genotype_length = alleles_count.length;

    // Select random allele of each gene
    int[] random_positions = new int[genotype_length];
    for (int i = 0; i < genotype_length; i++)
      if (alleles_count[i] != 0)
        random_positions[i] = EvRandomizer.INSTANCE.nextInt(alleles_count[i]);

    // Use individual to fill up expressed genes in chromosome
    Boolean[] alleles = new Boolean[genotype_length];
    ListIterator<Boolean> alleles_iterator =
        individual.getAlleles().listIterator();
    ListIterator<Integer> genes_iterator = individual.getGenes().listIterator();

    while (genes_iterator.hasNext()) {
      int gene = genes_iterator.next();
      boolean allele = alleles_iterator.next();

      if (random_positions[gene] == 0)
        alleles[gene] = allele;
      else
        random_positions[gene]--;
    }

    return alleles;
  }


  /*
   * Gets first allele of each gene. @param individual - messy individual to get
   * alleles from @return array of first alleles
   */
  private Boolean[] getFirstAlleles(EvMessyBinaryVectorIndividual individual) {

    // Create expressed_genes length array
    int expressed_genes = individual.getGenotypeLength();
    Boolean[] first_alleles = new Boolean[expressed_genes];
    for (int i = 0; i < expressed_genes; i++)
      first_alleles[i] = null;

    ListIterator<Boolean> alleles_iterator =
        individual.getAlleles().listIterator();
    ListIterator<Integer> genes_iterator = individual.getGenes().listIterator();

    // For each first allele set to its gene
    while (genes_iterator.hasNext()) {

      int pos = genes_iterator.next().intValue();
      boolean allele = alleles_iterator.next().booleanValue();

      if (pos < 0 || pos >= expressed_genes)
        throw new IllegalStateException("Gene " + pos + " is not in range");

      if (first_alleles[pos] == null)
        first_alleles[pos] = allele;
    }

    return first_alleles;

  }


  /*
   * Gets underspecified genes. The underspecified genes have randomly choosen
   * allele value or they have allele from template if it is set. @param genes -
   * list of genes to fill up, there is null if the gene is underspecified
   * @return full list of genes with alleles, there is specified allele for each
   * gene
   */
  private void getUnderspecifiedGenes(Boolean[] genes) {

    int genotype_length = genes.length;

    // Use template or random values to fill up the underspecifed genes
    if (template != null) {
      for (int i = 0; i < genotype_length; i++)
        if (genes[i] == null)
          genes[i] = template[i];
    } else
      for (int i = 0; i < genotype_length; i++)
        if (genes[i] == null)
          genes[i] = EvRandomizer.INSTANCE.nextBoolean();
  }

}