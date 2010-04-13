package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Solution space for EvMessyBinaryVectorIndividual. There are implemented
 * various individual generation methods.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */
public class EvMessyBinaryVectorSpace implements
    EvSolutionSpace<EvMessyBinaryVectorIndividual> {

  private static final long serialVersionUID = -3913477245135315803L;

  protected EvObjectiveFunction<EvMessyBinaryVectorIndividual> objective_function; // Objective
                                                                                    // function
                                                                                    // of
                                                                                    // the
                                                                                    // solution
                                                                                    // space

  protected int vector_length; // Binary vector length in the solution space


  /**
   * Constructor, override constructor of EvMessySpace for binary vector space.
   * 
   * @param objective_function objective function of the solution space
   * @param length binary vector length in the solution space
   */
  public EvMessyBinaryVectorSpace(
      EvObjectiveFunction<EvMessyBinaryVectorIndividual> objective_function,
      int vector_length) {
    if (vector_length < 1)
      throw new IllegalArgumentException(
          "Length of vector must be positive number.");

    this.setObjectiveFuntion(objective_function);
    this.vector_length = vector_length;
  }


  /**
   * {@inheritDoc}
   */
  @SuppressWarnings("unchecked")
  public boolean belongsTo(EvMessyBinaryVectorIndividual individual) {

    if (individual.getGenotypeLength() != vector_length)
      return false;

    ArrayList<Integer> genes = individual.getGenes();
    for (int i = 0; i < individual.getChromosomeLength(); i++)
      if (genes.get(i) < 0 || genes.get(i) >= vector_length)
        return false;

    return true;
  }


  /**
   * {@inheritDoc}
   */
  public EvMessyBinaryVectorIndividual takeBackTo(
      EvMessyBinaryVectorIndividual individual) {
    if (this.belongsTo(individual))
      return individual;
    else
      return null;
  }


  /**
   * Returns number of genes specified by individuals which belong to this space
   * 
   * @returns number of genes expressed by individuals in this space
   */
  public int getVectorLength() {
    return vector_length;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvMessyBinaryVectorIndividual> getObjectiveFuntion() {
    return objective_function;
  }


  /**
   * {@inheritDoc}
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvMessyBinaryVectorIndividual> objective_function) {
    if (objective_function == null)
      throw new IllegalArgumentException("Objective function cannot be null");

    this.objective_function = objective_function;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvMessyBinaryVectorIndividual>> divide(int n) {
    return null;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvMessyBinaryVectorIndividual>> divide(int n,
      Set<EvMessyBinaryVectorIndividual> p) {
    return null;
  }


  /**
   * Generates random MessyBinaryVectorIndividual, with equiprobably allele
   * values.
   * 
   * @returns new random MessyBinaryVectorIndividual
   */
  public EvMessyBinaryVectorIndividual generateIndividual() {
    return generateIndividual(null);
  }


  /**
   * Generates random MessyBinaryVectorIndividual, that has full set of genes in
   * uniformly random permutation, allele values are choosen with fixed
   * probability or equiprobably if not specified.
   * 
   * @param probability_vector vector of probabilities that there is true allele
   *        value for the gene, if it's null that means there are equiprobably
   *        allele values.
   * @returns new random MessyBinaryVectorIndividual
   */
  public EvMessyBinaryVectorIndividual generateIndividual(
      double[] probability_vector) {
    if (probability_vector != null)
      if (probability_vector.length != vector_length)
        throw new IllegalArgumentException(
            "probability_vector must have vector length of the space");

    ArrayList<Integer> genes = new ArrayList<Integer>(vector_length);

    // fill genes in randomly permutation
    int[] p = EvRandomizer.INSTANCE.nextPermutation(vector_length);
    for (int i = 0; i < vector_length; i++)
      genes.add(p[i]);

    // generate allele values
    ArrayList<Boolean> alleles = new ArrayList<Boolean>(vector_length);
    if (probability_vector == null)
      for (int i = 0; i < vector_length; i++)
        alleles.add(EvRandomizer.INSTANCE.nextDouble() < 0.5);
    else
      for (int i = 0; i < vector_length; i++)
        alleles
            .add(EvRandomizer.INSTANCE.nextDouble() < probability_vector[genes
                .get(i)]);

    EvMessyBinaryVectorIndividual individual =
        new EvMessyBinaryVectorIndividual(vector_length, genes, alleles);
    individual.setObjectiveFunction(objective_function);

    return individual;
  }


  /**
   * Generate list of individuals with all combinations of alleles, every
   * individual has specified gene list. Number of individuals is copies * 2 ^
   * (length of gene list).
   * 
   * @param genes gene list for all individuals
   * @param copies number of copies of each generated individual
   * @return list of individuals
   */
  @SuppressWarnings("unchecked")
  public List<EvMessyBinaryVectorIndividual> generateAllAlleleCombinationIndividuals(
      ArrayList<Integer> genes, int copies) {

    // Initialize
    int length = genes.size();

    int size = copies * 2;
    for (int i = 1; i < length; i++)
      size *= 2;

    ArrayList<EvMessyBinaryVectorIndividual> result =
        new ArrayList<EvMessyBinaryVectorIndividual>(size);

    // Create first allele combination
    ArrayList<Boolean> alleles = new ArrayList<Boolean>(length);
    for (int i = 0; i < length; i++)
      alleles.add(false);

    while (true) {
      // Create new individuals
      EvMessyBinaryVectorIndividual individual =
          new EvMessyBinaryVectorIndividual(vector_length,
              (ArrayList<Integer>) genes.clone(), (ArrayList<Boolean>) alleles
                  .clone());
      individual.setObjectiveFunction(objective_function);
      for (int i = 1; i < copies; i++)
        result.add(individual.clone());
      result.add(individual);

      // Create new combination (next lexicographic)
      int i;
      for (i = length - 1; i >= 0; i--)
        if (alleles.get(i).booleanValue())
          alleles.set(i, false);
        else {
          alleles.set(i, true);
          break;
        }

      if (i < 0)
        break;
    }

    return result;
  }


  /**
   * Generates list of individuals with all possible combinations of genes and
   * specified allele values. A combination of genes is a subset of all genes
   * (vector length). There is copies number which specify how many copies of
   * all individuals will be generated. <br>
   * Because the number od combinations can be extreme big, there can be
   * specified maximal number of individuals. If it is 0 that there is no limit,
   * else there will be equiprobably generated set of individuals with specified
   * size from set of all combination individuals. Copies are not affected to
   * this, so there will be generated (max_number * copies) individuals if
   * max_number is specified. If limit is more than all combinations number then
   * just all combinations are generated. <br>
   * Number of generated individuals is copies * C(vector_length, genes_number)
   * where C(n,k) = n! / (k! * (n-k)!) <br>
   * For example if vector_length = 4, genes_number = 2, copies = 1, then will
   * be generated indviduals which specify genes: (0,1) (0,2) (0,3) (1,2) (1,3)
   * (2,3). So there will be 6 individuals generated.
   * 
   * @param genes_number - number of specified genes in the generated
   *        individuals
   * @param copies - number of copies of generated individuals
   * @param max_individuals - maximal number of generated individuals, 0 means
   *        that there is no limit
   * @param alleles_template - alleles template of all possible genes for all
   *        generated individuals
   * @return list of individuals
   */
  @SuppressWarnings("unchecked")
  public List<EvMessyBinaryVectorIndividual> generateCoverIndividuals(
      int genes_number, int copies, int max_individuals,
      boolean[] alleles_template) {
    if (genes_number > vector_length)
      throw new IllegalArgumentException(
          "Genes number cannot be larger than vector length");
    if (copies <= 0)
      throw new IllegalArgumentException("Copies must be at least 1");
    if (alleles_template == null)
      throw new IllegalArgumentException("Alleles template cannot be null");
    if (alleles_template.length != vector_length)
      throw new IllegalArgumentException(
          "Alleles template must have vector length size");

    // Calculate size - number of combinations
    int size = 1;
    for (int i = vector_length; i > vector_length - genes_number; i--)
      size *= i;
    for (int i = 2; i <= genes_number; i++)
      size /= i;

    // Initialize
    if (max_individuals <= 0 || max_individuals > size)
      max_individuals = size;

    ArrayList<Integer> genes = new ArrayList<Integer>(genes_number);
    for (int i = 0; i < genes_number; i++)
      genes.add(i);

    ArrayList<Boolean> alleles = new ArrayList<Boolean>(genes_number);
    for (int i = 0; i < genes_number; i++)
      alleles.add(alleles_template[i]);

    ArrayList<EvMessyBinaryVectorIndividual> result =
        new ArrayList<EvMessyBinaryVectorIndividual>(size * copies);

    // *************************************************************************
    // Generate with alleles template

    while (true) {// All gene combinations loop
      /*
       * We choose with certain probability if we get this combination.
       */
      if (max_individuals >= size
          || EvRandomizer.INSTANCE.nextDouble() < (double) max_individuals
              / size) {

        // Create new individual
        EvMessyBinaryVectorIndividual individual =
            new EvMessyBinaryVectorIndividual(vector_length,
                (ArrayList<Integer>) genes.clone(),
                (ArrayList<Boolean>) alleles.clone());
        individual.setObjectiveFunction(objective_function);
        for (int i = 1; i < copies; i++)
          result.add(individual.clone());
        result.add(individual);

        max_individuals--;
      }

      size--;

      // Create new combination (lexicographic)
      int i;
      for (i = genes_number - 1; i >= 0; i--)
        if (genes.get(i).intValue() != vector_length - genes_number + i)
          break;

      if (i < 0)
        break;

      int v = genes.get(i);
      for (; i < genes_number; i++) {
        v++;
        genes.set(i, v);
        alleles.set(i, alleles_template[v]);
      }

    }

    // All combinations should be generated, so 0 is left
    assert (size == 0);
    // All individuals should be generated, so 0 is left
    assert (max_individuals == 0);

    return result;
  }


  /**
   * Generates list of individuals with all possible combinations of genes and
   * all possible combinations of allele values. A combination of genes and
   * alleles is a subset of all genes (vector length) and all allele values (<code>true, false</code>).
   * There is copies number which specify how many copies of all individuals
   * will be generated. <br>
   * Because the number od combinations can be extreme big, there can be
   * specified maximal number of individuals. If it is 0 that there is no limit,
   * else there will be equiprobably generated set of individuals with specified
   * size from set of all combination individuals. Copies are not affected to
   * this, so there will be generated (max_number * copies) individuals if
   * max_number is specified. If limit is more than all combinations number then
   * just all combinations are generated. <br>
   * Number of generated individuals is copies * C(vector_length, genes_number) *
   * 2^genes_number. <br>
   * For example if vector_length = 4, genes_number = 2, copies = 1, then will
   * be generated indviduals which specify genes: (0,1) (0,2) (0,3) (1,2) (1,3)
   * (2,3). For each of them, there will be 4 individuals which specify all
   * allele combinations: (0,0) (0,1) (1,0) (1,1). So there will be 6 * 4 = 24
   * individuals generated.
   * 
   * @param genes_number number of specified genes in the generated individuals
   * @param copies number of copies of generated individuals
   * @param max_individuals maximal number of generated individuals, 0 means
   *        that there is no limit
   * @return list of individuals
   */
  @SuppressWarnings("unchecked")
  public List<EvMessyBinaryVectorIndividual> generateCoverIndividuals(
      int genes_number, int copies, int max_individuals) {
    if (genes_number > vector_length)
      throw new IllegalArgumentException(
          "Genes number cannot be larger than vector length");
    if (copies <= 0)
      throw new IllegalArgumentException("Copies must be at least 1");

    // Calculate size - number of combinations
    int size = 1;
    for (int i = vector_length; i > vector_length - genes_number; i--)
      size *= i;
    for (int i = 2; i <= genes_number; i++)
      size /= i;
    for (int i = 0; i < genes_number; i++)
      size *= 2;

    // Initialize
    if (max_individuals <= 0 || max_individuals > size)
      max_individuals = size;

    ArrayList<Boolean> alleles = new ArrayList<Boolean>(genes_number);

    ArrayList<Integer> genes = new ArrayList<Integer>(genes_number);
    for (int i = 0; i < genes_number; i++)
      genes.add(i);

    for (int i = 0; i < genes_number; i++)
      alleles.add(false);

    ArrayList<EvMessyBinaryVectorIndividual> result =
        new ArrayList<EvMessyBinaryVectorIndividual>(size * copies);

    // *************************************************************************
    // Generate without template - all allele combinations

    while (true) { // All gene combinations loop
      while (true) { // All allele combinations loop

        /*
         * We choose with certain probability if we get this combination.
         */
        if (max_individuals >= size
            || EvRandomizer.INSTANCE.nextDouble() < (double) max_individuals
                / size) {

          // Create new individuals
          EvMessyBinaryVectorIndividual individual =
              new EvMessyBinaryVectorIndividual(vector_length,
                  (ArrayList<Integer>) genes.clone(),
                  (ArrayList<Boolean>) alleles.clone());
          individual.setObjectiveFunction(objective_function);
          for (int i = 1; i < copies; i++)
            result.add(individual.clone());
          result.add(individual);

          max_individuals--;
        }

        size--;

        // Create new combination (next lexicographic)
        int i;
        for (i = genes_number - 1; i >= 0; i--)
          if (alleles.get(i).booleanValue())
            alleles.set(i, false);
          else {
            alleles.set(i, true);
            break;
          }

        if (i < 0)
          break;
      }

      // Create new combination (lexicographic)
      int i;
      for (i = genes_number - 1; i >= 0; i--)
        if (genes.get(i).intValue() != vector_length - genes_number + i)
          break;

      if (i < 0)
        break;

      int v = genes.get(i);
      for (; i < genes_number; i++) {
        v++;
        genes.set(i, v);
      }
    }

    // All combinations should be generated, so 0 is left
    assert (size == 0);
    // All individuals should be generated, so 0 is left
    assert (max_individuals == 0);

    return result;
  }

}