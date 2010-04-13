package pl.wroc.uni.ii.evolution.distribution.strategies.topologies;

import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Base class for all distribution topologies. <br>
 * It contains basic strategy for assigning node to creation cell:
 * <ul>
 * <li> creation cell for node is chosen randomly
 * <li> creation cell for node is constant
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */

public abstract class EvTopology implements EvTopologyAssigner {

  private long creation_cell_id;

  private boolean creation_cell_computed = false;

  private long[] neighbours = null;


  /**
   * {@inheritDoc}
   */
  public long assignCellID() {
    if (!creation_cell_computed) {
      creation_cell_computed = true;
      creation_cell_id = EvRandomizer.INSTANCE.nextInt(getCellsCount());
    }
    return creation_cell_id;
  }


  /**
   * {@inheritDoc}
   */
  public long[] getNeighbours() {
    if (neighbours == null) {
      neighbours = getNeighbours(assignCellID());
    }
    return neighbours;
  }


  /**
   * @return number of cells in topology
   */
  public abstract int getCellsCount();


  /**
   * @param cell_id cell's indentifier
   * @return neighbors identifiers of given cell
   */
  public abstract long[] getNeighbours(long cell_id);

}
