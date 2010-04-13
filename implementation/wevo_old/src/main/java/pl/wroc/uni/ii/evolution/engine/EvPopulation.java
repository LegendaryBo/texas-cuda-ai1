package pl.wroc.uni.ii.evolution.engine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Base class for population of individuals
 * 
 * @author Piotr Baraniak, Donata Malecka
 */
public class EvPopulation<T extends EvIndividual> extends ArrayList<T> {

  /**
   * 
   */
  private static final long serialVersionUID = -7529593672829260773L;

  /**
   * This is population ID, which is called "hash" to mislead. It is a random
   * long value.
   */
  private long hash = 0L;

  /**
   * This variable determine if hash has to be validate in next call of
   * getHash() method.
   */
  private boolean hash_invalidated = true;


  /**
   * Make new empty Population with correct hash.
   */
  public EvPopulation() {
    super();
    hash = EvRandomizer.INSTANCE.nextLong();
    hash_invalidated = false;
  }


  /**
   * @param list List of individuals, which we want to have in our Population.
   */
  public EvPopulation(List<T> list) {
    super(list);
    hash = EvRandomizer.INSTANCE.nextLong();
    hash_invalidated = false;
  }


  /**
   * This constructor is because of cloning. We want to have two populations
   * with exact copies of individuals and same hash value. We use it in clone()
   * method.
   * 
   * @param parent Population which we want to clone.
   */
  @SuppressWarnings("unchecked")
  public EvPopulation(EvPopulation<T> parent) {
    super(parent);
    this.clear();
    for (int i = 0; i < parent.size(); i++) {
      this.add((T) parent.get(i).clone());
    }
    hash = parent.getHash();
    hash_invalidated = false;
  }


  public EvPopulation(int i) {
    super(i);
    hash = EvRandomizer.INSTANCE.nextLong();
    hash_invalidated = false;
  }


  public EvPopulation(T[] individuals) {
    this(Arrays.asList(individuals));
  }


  /**
   * @return Population ID.
   */
  public long getHash() {
    if (hash_invalidated) {
      hash = EvRandomizer.INSTANCE.nextLong();
      hash_invalidated = false;
    }
    return hash;
  }


  /*
   * (non-Javadoc) here go functions, that we need to override in order to know
   * when hash invalidates For all methods below,
   * 
   * @see java.util.ArrayList#add(int, java.lang.Object)
   */

  @Override
  public void add(int index, T element) {
    hash_invalidated = true;
    super.add(index, element);
  }


  @Override
  public boolean add(T o) {
    hash_invalidated = true;
    return super.add(o);
  }


  @Override
  public boolean addAll(Collection<? extends T> c) {
    hash_invalidated = true;
    return super.addAll(c);
  }


  @Override
  public boolean addAll(int index, Collection<? extends T> c) {
    hash_invalidated = true;
    return super.addAll(index, c);
  }


  @Override
  public T remove(int index) {
    hash_invalidated = true;
    return super.remove(index);
  }


  @Override
  public boolean remove(Object o) {
    hash_invalidated = true;
    return super.remove(o);
  }


  @Override
  protected void removeRange(int fromIndex, int toIndex) {
    hash_invalidated = true;
    super.removeRange(fromIndex, toIndex);
  }


  @Override
  public T set(int index, T element) {
    hash_invalidated = true;
    return super.set(index, element);
  }


  @Override
  public boolean removeAll(Collection<?> arg0) {
    hash_invalidated = true;
    return super.removeAll(arg0);
  }


  @Override
  public boolean retainAll(Collection<?> arg0) {
    hash_invalidated = true;
    return super.retainAll(arg0);
  }


  @Override
  public EvPopulation<T> clone() {
    getHash();
    return new EvPopulation<T>(this);
  }


  /**
   * This method sort Population and keep hash.
   */
  @SuppressWarnings("unchecked")
  public void sort() {
    long temp_hash = hash;
    Collections.sort(this);
    hash = temp_hash;
    hash_invalidated = false;
  }


  /**
   * This method reverse Population and keep hash.
   */
  public void reverse() {
    long temp_hash = hash;
    Collections.reverse(this);
    hash = temp_hash;
    hash_invalidated = false;
  }


  /**
   * @return Individual with maximum value of objective function in population.
   */
  public T getBestResult() {
    if (this.size() == 0) {
      return null;
    }

    T best_individual = this.get(0);

    for (T individual : this) {
      if (best_individual.compareTo(individual) < 0)
        best_individual = individual;
    }

    return best_individual;
  }


  /**
   * @return Individual with minimum value of objective function in population.
   */
  public T getWorstResult() {
    if (this.size() == 0) {
      return null;
    }

    T worst_individual = this.get(0);

    for (T individual : this) {
      if (worst_individual.compareTo(individual) > 0)
        worst_individual = individual;
    }

    return worst_individual;
  }


  /**
   * This method make subPopulation from population.
   */
  public EvPopulation<T> subList(int from, int to) {
    return new EvPopulation<T>(super.subList(from, to));
  }


  /**
   * This method set same objective function to all individuals in population.
   * 
   * @param function function which we want to optimize.
   */
  public void setObjectiveFunction(EvObjectiveFunction<T> function) {
    for (EvIndividual indiv : this)
      indiv.setObjectiveFunction(function);
  }


  /**
   * Invalidates hash for the population. Use this function in every operator
   * which modifies value of any of individuals contained in the population.
   */
  public void invalidateHash() {
    hash_invalidated = true;
  }


  public EvPopulation<T> kBest(int output_population_size) {
    EvKBestSelection<T> k_selection_children =
        new EvKBestSelection<T>(output_population_size);
    EvPopulation<T> apply = k_selection_children.apply(this);
    return apply;
  }
}
