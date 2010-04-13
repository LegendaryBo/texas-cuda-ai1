package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.bayesian.EvBayesianNetworkStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvBayesianOperator;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Operator for MIMIC algorithm. Operate above EvBinaryIndividuals.
 * 
 * @author Marek Chrusciel (bse@gmail.com)
 * @author Michal Humenczuk
 */

public class EvBinaryVectorMIMICOperator implements
    EvOperator<EvBinaryVectorIndividual>, EvBayesianOperator {

  /**
   * Size of returned population, given in constructor.
   */
  private int result_population_size;

  /**
   * Length of individuals in population.
   */
  private int dimension;

  /**
   * Stores permutation of columnn's indices after entrophy evaluation. 
   * Sort by conditional entrophies.
   */
  private int[] permutation;

  /**
   * Probability of getting <code>one</code> first element from
   * <code>permutation</code>.
   */
  private double first_permutation_element_one_probability;

  /**
   * table of conditional probabilities of getting <code>one</code> if
   * <code>one</code> was chosen before.
   */
  private double[] one_probabilities_vector;

  /**
   * table of conditional probabilities of getting <code>one</code> if
   * <code>zero</code> was chosen before.
   */
  private double[] zero_probabilities_vector;

  /**
   * Object in which we store bits dependances.
   */
  private EvPersistentStatisticStorage storage;


  /**
   * @param result_size of returned population
   */
  public EvBinaryVectorMIMICOperator(final int result_size) {
    result_population_size = result_size;
  }


  /**
   * Returns vector of booleans contains bits from column from population on
   * <code>column_index</code>.
   * 
   * @param population Input population
   * @param column_index Index of column
   * @return vector of booleans contains bits from column from population on
   *         <code>column_index</code>
   */
  protected Vector<Boolean> getColumn(
      final EvPopulation<EvBinaryVectorIndividual> population, 
      final int column_index) {
    Vector<Boolean> v = new Vector<Boolean>(population.size());

    int length = population.size();

    for (int i = 0; i < length; i++) {
      v.add(population.get(i).getGene(column_index) == 1);
    }

    return v;
  }


  /**
   * Evaluates entrophy of given vector.
   * 
   * @param column Column from population to evaluate
   * @return Evaluated entrophy of given vector
   */
  protected double getEntrophy(final Vector<Boolean> column) {
    double result = 0;
    int count_ones = 0;
    /** Counts 'ones' in given column */
    int vector_size = column.size();

    for (int i = 0; i < vector_size; i++) {
      if (column.get(i)) {
        count_ones++;
      }
    }

    /** Evaluate probability of getting 'one' */
    double prob_one = (double) count_ones / (double) column.size();

    /** Entrophy equals 0.0 if probability is 1.0 or 0.0 */
    if (prob_one == 1.0 || prob_one == 0.0) {
      return Double.MAX_VALUE;
    }
    /** Evaluate entrophy */
    result =
        -prob_one * Math.log(prob_one) - (1 - prob_one)
            * Math.log(1 - prob_one);
    return result;

  }


  /**
   * Evaluates conditional entrophy of column Y under condition from X column.
   * 
   * @param x Condition column
   * @param y Column to evaluate conditional entrophy
   * @return Conditional entrophy of column Y under condition from X column
   */
  protected double getConditionalEntrophy(
      final Vector<Boolean> x, final Vector<Boolean> y) {
    Vector<Boolean> ones = new Vector<Boolean>();
    Vector<Boolean> zeroes = new Vector<Boolean>();
    int count_ones = 0;

    /**
     * Split column Y by spoted bits (ones or zeroes) in column X. Count ones in
     * X also.
     */
    int x_size = x.size();

    for (int i = 0; i < x_size; i++) {
      if (x.get(i)) {
        count_ones++;
        ones.add(y.get(i));
      } else {
        zeroes.add(y.get(i));
      }
    }
    /** Evaluate probabilty of getting 'one' in X column */
    double prob_one = count_ones / x.size();

    /**
     * If X column contains only 'ones' or only 'zeroes' then special evaluate
     * is correct
     */
    if (ones.isEmpty()) {
      return (1 - prob_one) * getEntrophy(zeroes);
    }
    if (zeroes.isEmpty()) {
      return prob_one * getEntrophy(ones);
    }

    /** return evaluated conditional entrophy */
    return prob_one * getEntrophy(ones) + (1 - prob_one) * getEntrophy(zeroes);

  }


  /**
   * Gets index of column with maximal entrophy. If more than one column have
   * the same max entrophy factor random index from them is returned.
   * 
   * @param population Input population
   * @return index of column with maximal entrophy
   */
  protected int getColumnWithMaxEntrophy(
      final EvPopulation<EvBinaryVectorIndividual> population) {
    /** Dimension of individuals in population */
    int ind_dimension = population.get(0).getDimension();
    /** Create list of column with the same entrophy factor equals maximal */
    ArrayList<Integer> indexes = new ArrayList<Integer>();
    /** Get entrophy from first column as maximal (temporary) */
    indexes.add(new Integer(0));
    double max = getEntrophy(getColumn(population, 0));

    /** Find column with maximal entrophy */
    for (int i = 1; i < ind_dimension; i++) {
      double tmp = getEntrophy(getColumn(population, i));
      if (max == tmp) {
        indexes.add(new Integer(i));
      } else if (max < tmp) {
        max = tmp;
        indexes.clear();
        indexes.add(new Integer(i));
      }
    }
    Random r = new Random();
    return indexes.get(r.nextInt(indexes.size()));
  }


  /**
   * Gets index of column with maximal conditional entrophy. If more than one
   * column have the same max conditional entrophy factor, random index from
   * them is returned.
   * 
   * @param population Input popualtion
   * @param cond_column_index Conditional columnt index
   * @param unchecked List of column's indices to check with
   *        <code>cond_column_index</code>. Cannot be empty.
   * @return Index of column with maximal conditional entrophy
   */
  protected int getColumnWithMaxConditionalEntrophy(
      final EvPopulation<EvBinaryVectorIndividual> population,
      final Integer cond_column_index, final ArrayList<Integer> unchecked) {
    /** Create list of column with the same entrophy factor equals maximal */
    ArrayList<Integer> indexes = new ArrayList<Integer>();
    /** Get entrophy from first column as maximal (temporary) */
    indexes.add(unchecked.get(0));
    double max =
        getConditionalEntrophy(getColumn(population, cond_column_index),
            getColumn(population, unchecked.get(0)));
    /** Find column with maximal entrophy */
    for (int i = 1; i < unchecked.size(); i++) {
      double tmp =
          getConditionalEntrophy(getColumn(population, cond_column_index),
              getColumn(population, unchecked.get(i)));

      if (max == tmp) {
        indexes.add(unchecked.get(i));
      } else if (max < tmp) {
        max = tmp;
        indexes.clear();
        indexes.add(unchecked.get(i));
      }
    }

    Random r = new Random();

    return indexes.get(r.nextInt(indexes.size()));
  }


  /**
   * Evaluates permutation of columnn's indices sorted by conditional
   * entrophies.
   * 
   * @param population Input population
   */
  protected void evaluateEntrophiesPermutation(
      final EvPopulation<EvBinaryVectorIndividual> population) {
    /** Dimension of individuals in population */
    int ind_dimension = population.get(0).getDimension();
    /**
     * None of columns were checked, so 'unchecked' contains indices of all of
     * them
     */
    ArrayList<Integer> unchecked = new ArrayList<Integer>(ind_dimension);
    for (int i = 0; i < ind_dimension; i++) {
      unchecked.add(i);
    }
    /** Table to return */
    int[] result = new int[ind_dimension];

    /** Set first column by maximal entrophy */
    result[0] = getColumnWithMaxEntrophy(population);
    unchecked.remove((Object) result[0]);

    /** Evaluate conditional entrophy for column from unchecked */
    for (int i = 1; i < ind_dimension - 1; i++) {
      result[i] =
          getColumnWithMaxConditionalEntrophy(population, i - 1, unchecked);
      unchecked.remove((Object) result[i]);
    }

    /** Set last one element in permutation */
    result[ind_dimension - 1] = unchecked.get(0);
    /** Set permutation */
    this.permutation = result;
  }


  /**
   * Initializes vectors with conditional probabilities.
   * 
   * @param population Input population
   */
  protected void initPropabilitiesVector(
      final EvPopulation<EvBinaryVectorIndividual> population) {
    /** Set dimension of individuals in population */
    dimension = population.get(0).getDimension();
    /** Create vectors with conditional probabilities */
    one_probabilities_vector = new double[dimension];
    zero_probabilities_vector = new double[dimension];

    /**
     * Create counter and count number of 'ones' in first column from
     * permutation
     */
    int first_column_one_counter = 0;
    for (Boolean bit : getColumn(population, permutation[0])) {
      if (bit) {
        first_column_one_counter++;
      }
    }

    /** Probability of getting 'one' in firat column */
    first_permutation_element_one_probability =
        (double) first_column_one_counter / (double) population.size();

    /** Count 'zeroes' and 'ones' in each column under conditions */
    for (int i = 1; i < permutation.length; i++) {
      int one_cond_zero_counter = 0;
      int one_cond_one_counter = 0;
      int ones_amount = 0;
      int zeroes_amount = 0;

      Vector<Boolean> actual_column = getColumn(population, permutation[i]);
      Vector<Boolean> condition_column =
          getColumn(population, permutation[i - 1]);

      for (int j = 0; j < population.size(); j++) {
        if (condition_column.get(j)) {

          ones_amount++;

          if (actual_column.get(j)) {
            one_cond_one_counter++;
          }
        }
        if (!condition_column.get(j)) {
          zeroes_amount++;
          if (actual_column.get(j)) {
            one_cond_zero_counter++;
          }
        }
      }

      /** Evaluate probabilities */
      if (one_cond_one_counter == 0) {
        one_probabilities_vector[i] = 0;
      } else {
        one_probabilities_vector[i] =
            (double) one_cond_one_counter / (double) ones_amount;
      }

      if (one_cond_zero_counter == 0) {
        zero_probabilities_vector[i] = 0;
      } else {
        zero_probabilities_vector[i] =
            (double) one_cond_zero_counter / (double) zeroes_amount;
      }
    }
  }


  /**
   * Generate single individual in MIMIC way.
   * 
   * @param function Binary Individual objective function
   * @return new generated BinaryIndividual
   */
  protected EvBinaryVectorIndividual generateIndividual(
      final EvObjectiveFunction<EvBinaryVectorIndividual> function) {
    EvBinaryVectorIndividual ind = new EvBinaryVectorIndividual(dimension);
    ind.setObjectiveFunction(function);

    ind.setGene(permutation[0], EvRandomizer.INSTANCE
        .nextProbableBooleanAsInt(first_permutation_element_one_probability));

    /** Set bits in new individual with evaluated probabilities */
    for (int i = 1; i < dimension; i++) {
      if (ind.getGene(permutation[i - 1]) == 1) {
        ind.setGene(permutation[i], EvRandomizer.INSTANCE
            .nextProbableBooleanAsInt(one_probabilities_vector[i]));
      } else {
        ind.setGene(permutation[i], EvRandomizer.INSTANCE
            .nextProbableBooleanAsInt(zero_probabilities_vector[i]));
      }
    }
    return ind;
  }


  /**
   * {@inheritDoc}
   */
  @SuppressWarnings("unchecked")
  public EvPopulation<EvBinaryVectorIndividual> apply(
      final EvPopulation<EvBinaryVectorIndividual> population) {
    EvObjectiveFunction<EvBinaryVectorIndividual> function =
        population.get(0).getObjectiveFunction();

    /** Create new population */
    EvPopulation<EvBinaryVectorIndividual> new_population =
        new EvPopulation<EvBinaryVectorIndividual>(result_population_size);

    /**
     * Evaluate all needed factors to generate new individual. Check above
     * comments
     */
    evaluateEntrophiesPermutation(population);
    initPropabilitiesVector(population);

    /** Generate all individuals to new population */
    for (int i = 0; i < result_population_size; i++) {
      new_population.add(generateIndividual(function));
    }

    // saving statistics about bit dependeces
    if (storage != null) {
      
      int[][] edges = new int[permutation.length][2];
      
      for (int i = 1; i < permutation.length; i++) {
        edges[i][0] = permutation[i - 1];
        edges[i][1] = permutation[i];
      }
      
      EvBayesianNetworkStatistic stat = 
        new EvBayesianNetworkStatistic(edges, permutation.length);
      
      storage.saveStatistic(stat);
    }
    
    
    
    return new_population;
  }


  /**
   * {@inheritDoc}
   */
  public void collectBayesianStats(
      final EvPersistentStatisticStorage storage_) {
    storage = storage_;
  }

}
