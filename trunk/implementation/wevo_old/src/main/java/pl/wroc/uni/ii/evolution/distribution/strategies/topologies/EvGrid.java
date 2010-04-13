package pl.wroc.uni.ii.evolution.distribution.strategies.topologies;

import java.util.ArrayList;

/**
 * Grid topology n x m.<br>
 * Every node (except left, right, top and bottom) has 4 neighbors.
 * 
 * <pre>
 * |--|
 * ++++
 * ++++
 * ++++
 * |--|
 * </pre>
 * 
 * @author Marcin Golebiowski
 */

public class EvGrid extends EvTopology {

  private int n;

  private int m;


  /**
   * @param n height of grid (must be n >= 2)
   * @param m width of grid (must be m >= 2)
   */
  public EvGrid(int n, int m) {
    this.n = n;
    this.m = m;

    if (n < 2 || m < 2) {
      throw new IllegalArgumentException();
    }
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public int getCellsCount() {
    return n * m;
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public long[] getNeighbours(long creation_cell_id) {
    long top = creation_cell_id - m;
    long bottom = creation_cell_id + m;
    long left = creation_cell_id - 1;
    long right = creation_cell_id + 1;

    ArrayList<Long> result = new ArrayList<Long>();

    // check if creation cell has top neighbor
    if (top >= 0) {
      result.add(top);
    }

    // check if creation cell has bottom neighbor
    if (bottom <= n * m - 1) {
      result.add(bottom);
    }

    // check if creation cell has left neighbor
    if ((left / m) == (creation_cell_id / m) && (left >= 0)) {
      result.add(left);
    }

    // check if creation cell has left neighbor
    if ((right / m) == (creation_cell_id / m)) {
      result.add(right);
    }

    // copy to array
    long[] result_tab = new long[result.size()];
    for (int i = 0; i < result.size(); i++) {
      result_tab[i] = result.get(i);
    }

    return result_tab;
  }

}
