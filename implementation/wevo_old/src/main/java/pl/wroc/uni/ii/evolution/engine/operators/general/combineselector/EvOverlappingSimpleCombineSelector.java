package pl.wroc.uni.ii.evolution.engine.operators.general.combineselector;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCombineParentSelector;

/**
 * Implementation of overlapping EvCombineParentSelector used in crossover
 * operators.<br>
 * Just like EvNonOverlappingSimpleCombineSelector it returns individuals from
 * the beginning of the list to the end of the list.<br>
 * The difference is, it start from the beginning if the end of the list is
 * reached.<br>
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the operator works on
 */
class EvOverlappingSimpleCombineSelector<T extends EvIndividual> implements
    EvCombineParentSelector<T> {

  // implementation of circle list used in combine selector
  class EvCircleList<S extends EvIndividual> extends ArrayList<S> {

    private int next = 0;

    private static final long serialVersionUID = 1L;


    // gets element and moves pointer forward
    public S getNext() {
      if (this.size() == 0) {
        return null;
      }
      int cur = next;
      next = (next + 1) % this.size();
      return this.get(cur);
    }


    // moves pointer to specified place
    public void setNext(int next) {
      this.next = next;
    }

  }

  private EvCircleList<T> population = null;

  private int parent_count;

  private int count;

  private int called;


  // returns next individuals in circle list and moves it's
  // pointer forward.
  public List<T> getNextParents() {
    List<T> result = new ArrayList<T>();

    if (called < count) {
      population.setNext(called % population.size());
      for (int i = 0; i < parent_count; i++) {
        result.add(population.getNext());
      }
      called++;
    }

    return result;
  }


  // returm empty list
  // TODO shouldn't it return some individuals in some cases?
  public List<T> getUnselectedIndividuals() {
    return new ArrayList<T>();
  }


  public void init(List<T> population, int parent_count, int count) {

    this.population = new EvCircleList<T>();
    this.population.addAll(population);
    this.count = count;
    this.called = 0;
    this.parent_count = parent_count;
  }
}
