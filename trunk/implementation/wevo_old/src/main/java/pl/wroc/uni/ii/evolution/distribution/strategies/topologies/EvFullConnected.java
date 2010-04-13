package pl.wroc.uni.ii.evolution.distribution.strategies.topologies;

/**
 * FullConnected topology. Every pair of creation cells are connected.
 * 
 * @author Marcin Golebiowski
 */
public class EvFullConnected extends EvTopology {

  private int n;


  /**
   * @param n number of creation nodes (n > 0)
   */
  public EvFullConnected(int n) {

    if (n <= 0) {
      throw new IllegalArgumentException(
          "Ring topology: n must be positive Integer");
    }
    this.n = n;
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public long[] getNeighbours(long creation_cell_id) {
    if (n == 1) {
      return null;
    }
    long[] neighbours = new long[n - 1];

    int index = 0;
    for (int cell = 0; cell < n; cell++) {
      if (cell != creation_cell_id) {
        neighbours[index] = cell;
        index++;
      }
    }
    return neighbours;
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public int getCellsCount() {
    return n;
  }

}
