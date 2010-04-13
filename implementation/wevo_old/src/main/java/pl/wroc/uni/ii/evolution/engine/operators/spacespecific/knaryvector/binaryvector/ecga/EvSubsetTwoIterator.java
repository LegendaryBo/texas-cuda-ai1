package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

/**
 * Iterator over all possible pairs of integer in range [0, max_value). Elements
 * in pair are diffrent.
 * 
 * @author Marcin Golebiowski, Krzysztof Sroka, Marek Chrusciel
 */
public class EvSubsetTwoIterator {

  private int[] next_sets;

  private int max_value;


  /**
   * @param max_value specify range
   */
  public EvSubsetTwoIterator(int max_value) {
    if (max_value > 0) {
      next_sets = new int[] {0, (1 % max_value)};

    } else {
      next_sets = new int[] {0, 0};
    }
    this.max_value = max_value;
  }


  /**
   * Gets the next pair
   * 
   * @return next pair
   */
  public EvPair next() {
    EvPair next = new EvPair(next_sets[0], next_sets[1]);
    incrementCounter();
    return next;
  }


  private void incrementCounter() {
    /** increment */
    next_sets[1] = (next_sets[1] + 1);

    /** if next_sets[1] flipped over, increment next_sets[0] and set next_sets[1] */
    if (next_sets[1] == max_value) {
      next_sets[0] = (next_sets[0] + 1);
      next_sets[1] = (next_sets[0] + 1);
    }

    /** if second counter flipped over once more, finish */
    if (next_sets[1] == max_value) {
      next_sets[0] = 0;
      next_sets[1] = 0;
    }
  }


  /**
   * Check if there is a new pair to generate
   * 
   * @return
   */
  public boolean hasNext() {
    return (next_sets[0] != 0 || next_sets[1] != 0);
  }
}
