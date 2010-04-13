package pl.wroc.uni.ii.evolution.distribution.strategies.topologies;

/**
 * Interface to all class used for assign node to cell and getting neighbours
 * identifiers of this cell.
 * 
 * @author Marcin Golebiowski
 */
public interface EvTopologyAssigner {
  /**
   * Assign node to cell
   * 
   * @return node's cell identifier
   */
  public long assignCellID();


  /**
   * @return Neighbours identifiers
   */
  public long[] getNeighbours();
}
