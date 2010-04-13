package pl.wroc.uni.ii.evolution.utils;

import java.util.Random;

/**
 * A utility class. Provides some additional operations to that of the Random.
 * <p>
 * Please avoid using the static instance. It makes code harder to test. Take
 * randomizer as a parameter to the constructor if you can (at least provide one
 * that does.
 * 
 * @author Marcin Golebiewski (xormus@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */

public class EvRandomizer extends Random implements EvIRandomizer {

  /**
   *
   */
  private static final long serialVersionUID = -5352528771281646967L;

  /**
   * Singleton for EvRandomizer.
   */
  public static final EvRandomizer INSTANCE = new EvRandomizer();


  /**
   * Creates new Randomizer instance with random seed.
   */
  public EvRandomizer() {
  }


  /**
   * Creates new Randomizer instance with given seed.
   * 
   * @param seed - seed as int
   */
  public EvRandomizer(final int seed) {
    super(seed);
  }


  /**
   * Creates new Randomizer instance with given seed.
   * 
   * @param seed - seed as long
   */
  public EvRandomizer(final long seed) {
    super(seed);
  }


  /**
   * Returns a next logical value with given probability of true.
   * 
   * @param probability - probability of choosing <code>true</code>
   * @return random logical value
   */
  public boolean nextProbableBoolean(final double probability) {
    return nextDouble() < probability;
  }


  /**
   * Returns a next integer value from [0,1] with given probability of 1.
   * 
   * @param probability - probability of choosing <code>1</code>
   * @return random int value from [0,1]
   */
  public int nextProbableBooleanAsInt(final double probability) {
    if (nextDouble() < probability) {
      return 1;
    }
    return 0;
  }


  /**
   * Returns a random integer value in the given interval [start, end) or
   * [start, end].
   * 
   * @param start - start of the interval
   * @param end - end of the interval
   * @param end_inclusive - specify if end of interval belongs to it
   * @return random int value in the given interval
   */
  public int nextInt(final int start, final int end, final boolean end_inclusive) {
    if (end_inclusive) {
      return nextInt(end - start + 1) + start;
    } else {
      return nextInt(end - start) + start;
    }

  }


  /**
   * Returns a random Integer in given interval [start, end).
   * 
   * @param start - start of interval (inclusive)
   * @param end - end of interval (exclusive)
   * @return random int - random int value in [start, end)
   */
  public int nextInt(final int start, final int end) {
    return nextInt(start, end, false);
  }


  /**
   * Returns an array of <code>count</code> distinct integers from given
   * interval <code>[start, end)</code>.
   * 
   * @param start - start of interval (inclusive)
   * @param end - end of interval (exclusive)
   * @param count - returned list's length
   * @return list of distinct integers of length <code>count</code>
   */
  public int[] nextIntList(final int start, final int end, final int count) {
    int[] list = new int[count];
    int interval_len = end - start;
    int k = 0;

    if (start > end) {
      throw new IllegalArgumentException("Interval's start greater than end");
    }

    boolean[] bool_list = nextBooleanList(interval_len, count);

    for (int i = 0; i < bool_list.length; i++) {
      if (bool_list[i]) {
        list[k++] = start + i;
      }
    }

    return list;
  }


  /**
   * Creates random array of <code>size</code> boolean values with exactly
   * <code>count</code> set to true.
   * 
   * @param size - returned array size
   * @param count - number of <code>true</code> values in returned array
   * @return array of <code>boolean</code>s with <code>count</code>
   *         uniformly distributed <code>true</code> values
   */
  public boolean[] nextBooleanList(final int size, final int count) {

    if (size < count) {
      throw new IllegalArgumentException(
          "Interval length greater than expected list's length");
    }

    /*
     * Every subset of count elements of size elements set is equiprobably.
     * Proof by the induction: For size = 1 it is true. If count = 1 then
     * element is choosen, otherwise is not. For size > 1, assuming that it is
     * true for smaller size. The first element is choosen with pbb count /
     * size. Then there is count - 1 elements of size - 1 to choose. By the
     * assuming, every other element will be choosen with p1 = (count - 1) /
     * (size - 1). The first element is not choosen with pbb (size - count) /
     * size. Then there is count elements of size - 1 to choose. By the
     * assuming, every other element will be choosen with p2 = count / (size -
     * 1). So the first element has choosen pbb count / size, and every other
     * has pbb = (count / size) * p1 + (size - count) / size * p2 = (count /
     * size) * (count - 1) / (size - 1) + (size - count) / size * count / (size -
     * 1) = (count - 1 + size - count) * count / size / (size - 1) = (size - 1) *
     * count / size / (size - 1) = count / size.
     */

    boolean[] list = new boolean[size];
    int left = count;

    for (int i = size; i > 0; i--) {
      if (EvRandomizer.INSTANCE.nextDouble() < (double) left / i) {
        left--;
        list[i - 1] = true;
      }
    }

    return list;
  }


  /**
   * Synonymous to nextBooleanList, but it returns integer values instead of
   * booleans (<code>1 = true</code>, <code>0 = false</code>).
   * 
   * @param size - returned array size
   * @param count - number of <code>1</code> values in returned array
   * @return array of <code>integers</code>s with <code>count</code>
   *         uniformly distributed <code>1</code> values
   */
  public int[] nextBooleanListAsInt(final int size, final int count) {

    if (size < count) {
      throw new IllegalArgumentException(
          "Interval length greater than expected list's length");
    }

    int[] list = new int[size];
    int left = count;

    // Synonymous as in nextBooleanList
    for (int i = size; i > 0; i--) {
      if (EvRandomizer.INSTANCE.nextDouble() < (double) left / i) {
        left--;
        list[i - 1] = 1;
      }
    }

    return list;
  }


  /**
   * Returns a uniformly random permutation without retries, generated by Knuth
   * shuffle algorithm (The Art of Computer Programming vol. 2). The permutation
   * is represented as an array of length <code>n</code>, with distinct
   * values in the range <code>[0,n)</code>.
   * 
   * @param n - length of a permutation
   * @return permutation
   */
  public int[] nextPermutation(final int n) {
    int[] p = new int[n];

    // Set 1,...,n
    for (int i = 0; i < n; i++) {
      p[i] = i;
    }

    for (int i = n - 1; i > 0; i--) {
      // Swap with randomly chosen index
      int j = super.nextInt(i + 1);
      int swap = p[i];
      p[i] = p[j];
      p[j] = swap;
    }
    return p;
  }


  /**
   * Returns next random double number generated with normal (Gauss)
   * distribution with given parameters.
   * 
   * @param mean - mean (expected value) of generated numbers
   * @param std_deviation - standard deviation of generated numbers
   * @return double
   */
  public double nextGaussian(final double mean, final double std_deviation) {
    return std_deviation * super.nextGaussian() + mean;
  }

}
