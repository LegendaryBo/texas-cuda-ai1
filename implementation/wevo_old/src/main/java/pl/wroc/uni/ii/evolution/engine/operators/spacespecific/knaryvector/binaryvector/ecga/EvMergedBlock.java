package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

/**
 * Represents possible operation in ECGAStructure. Used in Cache. It contains
 * references to two Blocks and result of merge.
 * 
 * @author Marcin Golebiowski, Krzystof Sroka, Marek Chrusciel
 */
public class EvMergedBlock {

  /**
   * first block
   */
  private EvBlock first;

  /**
   * second block
   */
  private EvBlock second;

  /**
   * result block of merging the first and second block
   */
  private EvBlock result;


  public EvMergedBlock(EvBlock first, EvBlock second, EvBlock result) {
    this.result = result;
    this.first = first;
    this.second = second;
  }


  /**
   * Gets the first block
   * 
   * @return <code>first</code> block
   */
  public EvBlock getFirst() {
    return first;
  }


  /**
   * Gets the second block
   * 
   * @return <code>second</code> block
   */
  public EvBlock getSecond() {
    return second;
  }


  /**
   * Gets the result block
   * 
   * @return <code>result</code> block
   */
  public EvBlock getResult() {
    return result;
  }


  /**
   * Gets the profit of merging two blocks: <code>first</code> and
   * <code>second</code> into <code>result</code>
   * 
   * @return profit of merge
   */
  public double getProfit() {
    return (this.first.getRating() + this.second.getRating())
        - this.result.getRating();
  }


  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {

    if (first.getSize() + second.getSize() != result.getSize()) {
      throw new IllegalArgumentException("[(" + first + "{" + first.getRating()
          + "})+(" + second + "{" + second.getRating() + "})]=[" + result + "{"
          + result.getRating() + "," + getProfit() + "})]");
    }

    return "[(" + first + "{" + first.getRating() + "})+(" + second + "{"
        + second.getRating() + "})]=[" + result + "{" + result.getRating()
        + "," + getProfit() + "})]";
  }
}
