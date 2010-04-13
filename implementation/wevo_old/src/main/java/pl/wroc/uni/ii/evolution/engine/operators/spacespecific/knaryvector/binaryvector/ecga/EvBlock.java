package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Represents block of indexes in ECGA Structure.
 * 
 * @author Marcin Golebiowski, Krzysztof Sroka, Marek Chrusciel
 */
public class EvBlock {

  public ArrayList<Integer> values;

  /**
   * block rating
   */
  private double rating;


  /**
   * Default constructor
   */
  public EvBlock() {
    values = new ArrayList<Integer>();
  }


  /**
   * Add <code> index </code> to block
   * 
   * @param index
   */
  public void put(int index) {
    values.add(0, index);
  }


  /**
   * Gets current block size
   * 
   * @return block size
   */
  public int getSize() {
    return values.size();
  }


  /**
   * Gets block values in ArrayList
   * 
   * @return list of block elements
   */
  public ArrayList<Integer> getElements() {
    return new ArrayList<Integer>(values);
  }


  /**
   * Add all elements of <code> some_block </code> to current block
   * 
   * @param some_block block to merge wi
   */
  public void merge(EvBlock some_block) {
    values.addAll(some_block.getElements());
  }


  /**
   * Sets block rating
   * 
   * @param rating new rating for block
   */
  public void setRating(double rating) {
    this.rating = rating;
  }


  /**
   * Gets current block rating
   * 
   * @return current rating
   */
  public double getRating() {
    return this.rating;
  }


  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();

    sb.append("[");
    for (Integer el : values) {
      sb.append(el.toString());
      sb.append(" ");
    }
    sb.append("]");

    return sb.toString();
  }


  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#equals(java.lang.Object)
   */
  @Override
  public boolean equals(Object obj) {

    if (!(obj instanceof EvBlock)) {
      return false;
    }

    EvBlock b = (EvBlock) obj;

    ArrayList<Integer> el = b.getElements();
    Object[] bb = el.toArray();
    Object[] aa = values.toArray();

    Arrays.sort(aa);
    Arrays.sort(bb);

    if (bb.length != aa.length) {
      return false;
    }

    for (int i = 0; i < bb.length; i++) {
      if (bb[i] != aa[i]) {
        return false;
      }
    }

    return true;
  }
}
