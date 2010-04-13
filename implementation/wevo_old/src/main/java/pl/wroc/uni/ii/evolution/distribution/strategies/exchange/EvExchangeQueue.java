package pl.wroc.uni.ii.evolution.distribution.strategies.exchange;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Custom PriorityQueue. Queue can have at most <code> max_queue_size </code>
 * elements and can ensure that it hasn't duplicates.
 * 
 * @author Marcin Golebiowski
 */
public class EvExchangeQueue<T extends EvIndividual> extends PriorityQueue<T> {

  private static final long serialVersionUID = -93591780249183160L;

  private boolean allows_duplicates;

  private int max_queue_size;


  /**
   * @param allows_duplicates
   * @param max_queue_size
   */
  public EvExchangeQueue(boolean allows_duplicates, int max_queue_size) {
    super(10, new Comparator<T>() {
      public int compare(T o1, T o2) {
        return -o1.compareTo(o2);
      }
    });
    this.allows_duplicates = allows_duplicates;
    this.max_queue_size = max_queue_size;
  }


  @Override
  public boolean offer(T o) {

    if (!allows_duplicates && this.contains(o)) {
      return false;
    }

    super.offer(o);

    if (this.size() > max_queue_size) {
      List<T> tmp = new ArrayList<T>();
      for (int i = 0; i < max_queue_size; i++) {
        tmp.add(this.poll());
      }
      this.clear();
      this.addAll(tmp);
    }
    return true;
  }
}
