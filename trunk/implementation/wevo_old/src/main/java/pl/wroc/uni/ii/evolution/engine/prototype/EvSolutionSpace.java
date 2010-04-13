package pl.wroc.uni.ii.evolution.engine.prototype;

import java.io.Serializable;
import java.util.Set;

/**
 * Basic interface that every SoultionSpace must implement. Hint: Dividing space
 * into equal regions can be done using sampling random n elements from parent
 * space and then classifying an element to a specific subspace using nearest
 * neighbor method.
 * 
 * @author Marcin Brodziak (marcin@nierobcietegowdomu.pl)
 * @author Tomasz Kozakiewicz (quzzaq@gmail.com)
 * @param <T> - type of individuals represented by this solution space
 */

public interface EvSolutionSpace<T extends EvIndividual> extends Serializable {

  /**
   * Divides space into n subspaces of more-less the same size.
   * 
   * @param n - number of subspaces to generate
   * @return - set of generated subspaces
   */
  Set<EvSolutionSpace<T>> divide(int n);


  /**
   * Divides space into n subspaces using set of individuals as a hint for
   * generation (i.e. using k-means clustering).
   * 
   * @param n -- number of subspaces to generate
   * @param p -- a set of individuals
   * @return set of generated subspaces
   */
  Set<EvSolutionSpace<T>> divide(int n, Set<T> p);


  /**
   * Checks if individual belongs to the solution space.
   * 
   * @param individual -- checked individual
   * @return true if a given individual belongs to solution space, false
   *         otherwise
   */
  boolean belongsTo(T individual);


  /**
   * Set objective function to current solution space so that it can generate
   * individuals containing objective function.<BR>
   * Use this method after creation of solution space
   * 
   * @param objective_function - function that evaluate individuals
   */
  void setObjectiveFuntion(EvObjectiveFunction<T> objective_function);


  /**
   * @return the objective function of the solution space.
   */
  EvObjectiveFunction<T> getObjectiveFuntion();


  /**
   * Brings an individual back to the solution space by finding one of nearest
   * elements in solution space to the given individual. If individual belongs
   * to solution space returns it.
   * 
   * @param individual -- some individual
   * @return individual taken back to the solution space
   */
  T takeBackTo(T individual);


  /**
   * Generates a random individual from the solution space. Preferred
   * implementation should give a uniform probability of generating any
   * individual from a given space. We know that in some cases constructor may
   * perform some optimization and generate not uniformly random individuals
   * that are expected to give better results, however this assumption is not
   * used in construction of algorithms. Use common sense.<BR>
   * Every generated individual should have objective function set.
   * 
   * @return random individual from the solution space.
   */
  T generateIndividual();
}