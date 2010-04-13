package pl.wroc.uni.ii.evolution.distribution.strategies.topologies;

import java.util.BitSet;

/**
 * Implementation of HiperCube topology with n dimension.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvHyperCube extends EvTopology {

  private int dimension;


  /**
   * @param dimension hypercube's dimension (must be greater than zero and less
   *        then to 64).
   */
  public EvHyperCube(int dimension) {
    this.dimension = dimension;

    if (dimension <= 0 || dimension >= 64) {
      throw new IllegalArgumentException();
    }
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public int getCellsCount() {
    return (int) Math.pow(2.0, dimension);
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public long[] getNeighbours(long creation_cell_id) {

    long[] result = new long[dimension];

    BitSet bit_set = getBitSetFromLong(creation_cell_id);

    for (int i = 0; i < dimension; i++) {
      bit_set.flip(i);
      result[i] = getLongFromBitSet(bit_set);
      bit_set.flip(i);
    }

    return result;
  }


  /**
   * Returns BitSet that is binary representation of number
   * 
   * @param number
   * @return BitSet
   */
  public BitSet getBitSetFromLong(long number) {
    BitSet bit_vector = new BitSet(64);

    int index = 0;
    while (number != 0) {
      bit_vector.set(index, (number % 2 == 1));
      index++;
      number = number >> 1;
    }
    return bit_vector;
  }


  /**
   * Returns number with given binary representation
   * 
   * @param bit_set
   * @return long
   */
  public long getLongFromBitSet(BitSet bit_set) {
    long number = 0;

    for (int i = 0; i < bit_set.length(); i++) {
      if (bit_set.get(i)) {
        number += (long) Math.pow(2.0, i);
      }
    }
    return number;
  }

}
