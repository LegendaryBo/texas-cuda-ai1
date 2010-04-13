package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A solution space of EvMessyIndividuals
 * 
 * @author Krzysztof Sroka, Marcin Golebiowski
 */
public class EvSimplifiedMessySpace implements
    EvSolutionSpace<EvSimplifiedMessyIndividual> {

  private static final long serialVersionUID = -6913477245135315803L;

  private EvObjectiveFunction<EvSimplifiedMessyIndividual> objective_function;

  private int chromosome_length;

  private int max_value_of_gene;


  /**
   * Executes <code> MessySpace(chromosome_len, 1) </code>
   * 
   * @param objective_function objective function evaluated by generated
   *        individuals
   * @param chromosome_len accepted chromosome length
   */
  public EvSimplifiedMessySpace(
      EvObjectiveFunction<EvSimplifiedMessyIndividual> objective_function,
      int chromosome_len) {
    this(objective_function, chromosome_len, 1);
  }


  /**
   * Creates new solution space accepting MessyIndividuals with chromosomes of
   * length <code> chromosome_length </code> and genes values from zero up to
   * space_size
   * 
   * @param objective_function objective function evaluated by generated
   *        individuals
   * @param chromosome_length
   * @param max_value_of_gene maximum value of gene
   */
  public EvSimplifiedMessySpace(
      EvObjectiveFunction<EvSimplifiedMessyIndividual> objective_function,
      int chromosome_length, int max_value_of_gene) {

    // basic sanity checks
    if (max_value_of_gene < 1) {
      throw new IllegalArgumentException("Space size to low");
    }

    if (chromosome_length < 1) {
      throw new IllegalArgumentException("Chromosome length to low");
    }

    this.chromosome_length = chromosome_length;
    this.max_value_of_gene = max_value_of_gene;
    setObjectiveFuntion(objective_function);
  }


  /**
   * {@inheritDoc}
   */
  public boolean belongsTo(EvSimplifiedMessyIndividual individual) {

    // basic sanity check
    if (individual.getLength() != chromosome_length) {
      return false;
    }

    for (int i = individual.getLength() - 1; i >= 0; i--) {

      ArrayList<Integer> gen_values = individual.getGeneValues(i);

      if (gen_values.size() == 0) {
        continue;
      }

      Integer[] array_values =
          (Integer[]) convertObjectArray(gen_values.toArray());

      Arrays.sort(array_values);

      // search for duplicate values
      int prev = array_values[0];
      for (int j = 1; j < array_values.length; j++) {
        if (prev == array_values[j]) {
          return false;
        }
        prev = array_values[j];
      }
      // check if gen's values are in correct interval
      for (Integer gen_value : gen_values) {
        if (gen_value > max_value_of_gene) {
          return false;
        }
      }
    }
    return true;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvSimplifiedMessyIndividual>> divide(int n) {
    return null;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvSimplifiedMessyIndividual>> divide(int n,
      Set<EvSimplifiedMessyIndividual> p) {
    return null;
  }


  /**
   * Generates random MessyIndividual.
   */
  public EvSimplifiedMessyIndividual generateIndividual() {
    EvSimplifiedMessyIndividual mind =
        new EvSimplifiedMessyIndividual(chromosome_length);

    for (int i = chromosome_length - 1; i >= 0; i--) {
      mind.setGeneValue(i, EvRandomizer.INSTANCE.nextInt(0,
          max_value_of_gene + 1));
    }
    mind.setObjectiveFunction(objective_function);

    return mind;
  }


  /**
   * @return space size
   */
  public int getMaxValueOfGene() {
    return this.max_value_of_gene;
  }


  /**
   * {@inheritDoc}
   */
  public EvSimplifiedMessyIndividual takeBackTo(
      EvSimplifiedMessyIndividual individual) {
    return individual;
  }


  private Integer[] convertObjectArray(Object[] array) {
    Integer[] tab = new Integer[array.length];

    for (int i = 0; i < array.length; i++) {
      tab[i] = (Integer) array[i];
    }
    return tab;
  }


  /**
   * {@inheritDoc}
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvSimplifiedMessyIndividual> objective_function) {
    if (objective_function == null) {
      throw new IllegalArgumentException("Objective function cannot be null");
    }
    this.objective_function = objective_function;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvSimplifiedMessyIndividual> getObjectiveFuntion() {
    return objective_function;
  }
}
