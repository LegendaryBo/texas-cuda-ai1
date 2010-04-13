package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

import java.util.ArrayList;

/**
 * Class is used for improve speed of ECGA by storing set profits. Works like
 * cache.
 * 
 * @author Marcin Golebiowski
 */
public class EvCache {

  /**
   * Cache data
   */
  private ArrayList<EvMergedBlock> entries;


  /**
   * Default constructor
   */
  public EvCache() {
    entries = new ArrayList<EvMergedBlock>();
  }


  /**
   * Add <code> block </code> to cache
   * 
   * @param block merged block to add
   */
  public void put(EvMergedBlock block) {
    entries.add(block);
  }


  /**
   * Remove from cache all MergedBlocks <code> block </code> that have
   * <code> block.first = first </code> or <code> block.second = second </code>
   * 
   * @param first
   * @param second
   */
  public void remove(EvBlock first, EvBlock second) {
    for (int i = entries.size() - 1; i >= 0; i--) {
      if (entries.get(i).getFirst() == first
          || entries.get(i).getSecond() == second
          || entries.get(i).getFirst() == second
          || entries.get(i).getSecond() == first) {
        entries.remove(i);
      }
    }
  }


  /**
   * Get merged block with index <code>i</code>
   * 
   * @param i index
   * @return chosen merged block
   */
  public EvMergedBlock getMergedBlock(int i) {
    return entries.get(i);
  }


  /**
   * Gets cache size
   * 
   * @return amount of MergedBlocks in cache
   */
  public int getCacheSize() {
    return entries.size();
  }


  /**
   * Gets best merged block (block with max profit)
   * 
   * @return MergedBlock with the best profit
   */
  public EvMergedBlock getBestMerged() {

    if (entries.size() <= 0) {
      return null;
    }

    EvMergedBlock best = entries.get(0);
    double max_val = entries.get(0).getProfit();

    for (int i = 1; i < entries.size(); i++) {
      if (max_val < entries.get(i).getProfit()) {
        best = entries.get(i);
        max_val = best.getProfit();
      }
    }
    return best;
  }


  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();

    for (EvMergedBlock block : entries) {
      sb.append(block.toString());
      sb.append("\n");
    }
    return sb.toString();
  }
}
