package pl.wroc.uni.ii.evolution.distribution.strategies.topologies;

/**
 * Ring topology. There are n creation cells connected in circle.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvRing extends EvTopology {

  private int n;


  /**
   * @param n number of creation cells (n > 2)
   */
  public EvRing(int n) {

    if (n <= 2) {
      throw new IllegalArgumentException(
          "Ring topology: n must be positive Integer greater than <code> two </code>");
    }
    this.n = n;

  }


  /**
   * {@inheritDoc}
   */
  @Override
  public int getCellsCount() {
    return n;
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public long[] getNeighbours(long creation_cell_id) {
    long next;
    long prev;

    next = (creation_cell_id + 1) % n;

    if (creation_cell_id == 0) {
      prev = n - 1;
    } else {
      prev = (creation_cell_id - 1) % n;
    }

    if (next != prev) {
      return new long[] {prev, next};
    } else {
      return null;
    }
  }

}
