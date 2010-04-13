package pl.wroc.uni.ii.evolution.utils;

import java.util.HashSet;
import java.util.Set;

import junit.framework.TestCase;

/**
 * Tests for EvRandomizer random numbers generator. 
 * 
 * @author Marcin Golebiewski (xormus@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public class EvRandomGeneratorTest extends TestCase {

  /**
   * Tested EvRandomizer instance.
   */
  private EvRandomizer randomizer = new EvRandomizer(0);

  /**
   * Tests predictability of random numbers generator.
   */
  public void testNextIntSeededWithZero() {
    int firstIntegerReturnedByRandomSeededWithZero = -1155484576;
    assertEquals(firstIntegerReturnedByRandomSeededWithZero, randomizer
        .nextInt());
  }
  
  /** 
   * Tests whether nextProbableBoolean(1.0) always returns <code>true</code>.
   */
  public void testProbableBooleanAlwaysTrue() {
    randomizer = new EvRandomizer();

    for (int i = 0; i < 100; i++) {
      assertTrue(randomizer.nextProbableBoolean(1.0));
    }
  }

  /**
   * Tests whether nextProbableBoolean(0.0) always returns <code>false</code>.
   * @throws Exception
   */
  public void testProbableBooleanNoChanceForTrue() {
    randomizer = new EvRandomizer();

    for (int i = 0; i < 100; i++) {
      assertFalse(randomizer.nextProbableBoolean(0.0));
    }
  }

  /** 
   * Tests nextProbableBoolean(double) with non-trivial cases.
   */ 
  public void testProbableBooleanCustomChance() {
    int tries = 2000;
    float error = 0.01f;

    testNextProbableBooleanWorksAsExpected(0.25f, tries, error);
    testNextProbableBooleanWorksAsExpected(0.75f, tries, error);
  }

  /**
   * Helper function for testing the nextProbableBoolean(double) method.
   * 
   * @param chance parameter passed to <code>nextProbableBoolean</code>.
   * @param tries rounds of generation
   * @param error Acceptable error
   */
  private void testNextProbableBooleanWorksAsExpected(
      final float chance, final int tries, final float error) {
    randomizer = new EvRandomizer(0);
    int trueCount = 0;
    for (int i = 0; i < tries; i++) {
      if (randomizer.nextProbableBoolean(chance)) {
        trueCount++;
      }
    }

    assertEquals(chance, (float) trueCount / (float) tries, error);
  }

  /**
   * Tests whether <code>nextBooleanList</code> returns correct number of
   * <code>true</code> values.
   */
  public void testRandomBooleanListCorrectNumberOf1s() {
    for (int i = 0; i <= 30; i++) {
      assertArrayHasThatNumberOf1s(randomizer.nextBooleanList(30, i), i);
    }
  }

  /**
   * Helper function testing number of <code>true</code> values in array of
   * boolean values.
   * 
   * @param array tested array 
   * @param expected_count expected number of <code>true</code> values in the 
   *     given <code>array</code>
   */
  private void assertArrayHasThatNumberOf1s(final boolean[] array,
      final int expected_count) {
    int count = 0;
    for (boolean val : array) {
      if (val) {
        count++;
      }
    }

    assertEquals(expected_count, count);
  }

  /**
   * Generates a number of boolean arrays with only one true. The position of
   * true should be in the middle of the array length.
   */
  public void testRandomBooleanListGoodDistributionPosOfTrue() {

    int expected_position = 15;
    int tolerance = 2;
    int array_lenght = 31;

    testDistributionOfRandomBooleanListMeanPositionOf(true, expected_position,
        tolerance, array_lenght);
  }

  /**
   * Generate a number of boolean arrays with only one false. The position of
   * false should be in the middle of the array length.
   */
  public void testRandomBooleanListGoodDistributionPosOfFalse() {

    int expected_position = 15;
    int tolerance = 2;
    int array_lenght = 31;

    testDistributionOfRandomBooleanListMeanPositionOf(false,
        expected_position, tolerance, array_lenght);
  }

  /**
   * Tests mean position of selected value in generated list of booleans. 
   * 
   * A number of random boolean arrays are drawn. Arrays are specified so as to
   * have only one occurence of value (parameter to method). The mean position
   * of that value is computed and checked against expectation.
   * 
   * @param val Boolean value to test
   * @param expected_position Expected mean position of <code>val</code>
   * @param tolerance Upper bound of error in observed mean position
   * @param length length of generated arrays
   */
  private void testDistributionOfRandomBooleanListMeanPositionOf(
      final boolean val, final int expected_position, final int tolerance,
      final int length) {
    int cumulation = 0;
    final int iterations = 1000;
    randomizer = new EvRandomizer(0);
    int true_values_number;
    if (val) {
      true_values_number = 1;
    } else {
      true_values_number = length - 1;
    }
    
    for (int i = 0; i < iterations; i++) {
      boolean[] random_array =
          randomizer.nextBooleanList(length, true_values_number);
      cumulation += firstPositionOf(val, random_array);
    }

    double observed_mean_position = cumulation / (double) iterations;
    assertEquals(expected_position, observed_mean_position,
        tolerance);
  }

  /**
   * Helper function finding first occurrence of a given value in array.
   *
   * @param toFind value to look for
   * @param random_array randomly generated array to test
   * @return position of first occurrence of <code>toFind</code> in 
   *    <code>random_array</code> or <code>-1</code> if it does not occur.
   */
  private int firstPositionOf(final boolean toFind, 
      final boolean[] random_array) {
    for (int i = 0; i < random_array.length; i++) {
      if (random_array[i] == toFind) {
        return i;
      }
    }
    fail("Expected value not found in the given array");
    return -1;
  }

  /**
   * Tests distribution of values in randomly generated array of booleans.
   */
  public void testDistributionOfBooleanListGoodDistributionArrayOfMeans() {
    int array_length = 30;

    final int iterations = 2000;
    // expected value is in [0,1]
    double tolerance = 0.04; // 4% tolerance

    testDistributionOfBooleanListMeans(array_length, iterations, tolerance,
        array_length / 2);
    testDistributionOfBooleanListMeans(array_length, iterations, tolerance,
        10);
    testDistributionOfBooleanListMeans(array_length, iterations, tolerance,
        array_length - 3);
    testDistributionOfBooleanListMeans(array_length, iterations, tolerance,
        array_length - 2);
    testDistributionOfBooleanListMeans(array_length, iterations, tolerance,
        array_length - 1);
  }

  /**
   * Helper function to test uniform distribution of values in random boolean
   * list.
   * 
   * @param length length of generated arrays 
   * @param iterations number of generated arrays 
   * @param tolerance upper bound on error between expected and observed
   *    probability of values in generated arrays
   * @param true_values number of <code>true</code> values in
   *    generated arrays
   */
  private void testDistributionOfBooleanListMeans(final int length,
      final int iterations, final double tolerance, final int true_values) {
    int[] number_of_trues = new int[length];
    randomizer = new EvRandomizer(0);
    for (int i = 0; i < iterations; i++) {
      boolean[] random_array =
          randomizer.nextBooleanList(length, true_values);

      add(number_of_trues, random_array);
    }

    double expected_mean = (double) true_values / (double) length;
    for (int i = 0; i < length; i++) {
      assertEquals(i + "th position should be true with probability of "
          + expected_mean, expected_mean, (double) number_of_trues[i]
          / (double) iterations, tolerance);
    }
  }

  /**
   * Helper function to count occurrences of <code>true</code> valus in the
   * given array.
   * @param number_of_trues Array of occurrences count, updated to add 
   *    occurrences of <code>true</code> values in <code>random_array</code> 
   * @param random_array Randomly generated array of boolean values 
   */
  private void add(final int[] number_of_trues, final boolean[] random_array) {
    for (int j = 0; j < number_of_trues.length; j++) {
      if (random_array[j]) {
        number_of_trues[j] = number_of_trues[j] + 1;
      }
    }
  }

  /**
   * Tests length of randomly generated array of integer values. 
   */
  public void testIntListLength() {
    for (int i = 1; i < 12; i++) {
      assertEquals(i * i, randomizer.nextIntList(3, 800, i * i).length);
    }
  }
  
  /**
   * Tests whether randomly generated array of integer contains only specified
   * values.
   */
  public void testIntListValuesWithinRange() {
    int length = 25;
    int min = -12;
    int max_exclusive = 20;
    testValuesAreWithinRange(min, max_exclusive, length);
    testValuesAreWithinRange(Integer.MAX_VALUE - 300, Integer.MAX_VALUE, 299);
  }

  /**
   * Helper function to test whether randomly generated array of integer 
   * contains only specified values. 
   * @param min lower bound of expected values
   * @param max_exclusive upper bound (exclusive) of expected values
   * @param length length of randomly generated array
   */
  private void testValuesAreWithinRange(final int min, final int max_exclusive,
      final int length) {
    int[] ints = randomizer.nextIntList(min, max_exclusive, length);

    for (int val : ints) {
      assertTrue(val >= min && val < max_exclusive);
    }
  }

  /**
   * Tests whether randomly generated arrays of integers contain only unique
   * values.
   */
  public void testIntListDistinctValues() {
    assertHasDistinctValues(randomizer.nextIntList(0, 200, 10));
    assertHasDistinctValues(randomizer.nextIntList(-12, 12, 24));
    assertHasDistinctValues(randomizer.nextIntList(Integer.MAX_VALUE - 300,
        Integer.MAX_VALUE, 299));
  }

  /**
   * Helper function testing whether given array of integers contains only 
   * unique values.
   * @param int_array array to test
   */
  private void assertHasDistinctValues(final int[] int_array) {
    Set<Integer> set = new HashSet<Integer>();

    for (int val : int_array) {
      set.add(new Integer(val));
    }

    assertEquals(int_array.length, set.size());
  }

  /**
   * Tests if randomizer fails to create random array of integers when list 
   * length is greater than interval of expected (unique) vales.  
   */
  public void testIntListIntervalSmallerThanListLength() {
    try {
      randomizer.nextIntList(0, 12, 13);
      fail("Exception expected");
    } catch (IllegalArgumentException e) {
      return; // ok
    } catch (Exception e) {
      fail("IllegalArgumentException expected");
    }
  }

  /**
   * Tests if randomizer fails to create an array of integer with upper bound
   * smaller than lower bound. 
   */
  public void testIntListIntervalStartGraterThanEnd() {
    try {
      randomizer.nextIntList(20, 12, 4);
      fail("Exception expected");
    } catch (IllegalArgumentException e) {
      return; // ok
    } catch (Exception e) {
      fail("IllegalArgumentException expected");
    }
  }

  /**
   * Tests whether permutation generator creates array of unique values in a 
   * given interval.
   */
  public void testNextPermutation() {
    int n = 20; // Permutation size to test
    int[] p = randomizer.nextPermutation(n);

    assertEquals(p.length, n);

    // Check if all values are distinct and in the range [0,n)
    boolean[] v = new boolean[n];
    for (int i = 0; i < n; i++) {
      if (p[i] >= 0 && p[i] < n) {
        assertFalse(v[p[i]]);
        if (v[p[i]]) {
          fail("Values in the permutation are not distinct");
        } else {
          v[p[i]] = true;
        }
      } else {
        fail("Values in the permutation is out of range");
      }
    }
  }

  /**
   * Tests nextGaussian(double, double) method by creating some ammount of
   * randomly generated numbers and comparing results with expected values
   * (parameters for generation). Fails if the error in generated values is too
   * high.
   */
  public void testNextGaussian() {
    // parameters for nextGaussian(...) method
    double generated_mean = 1337.0;
    double generated_deviation = 1000.0;
    // observed results
    double observed_mean = 0.0;
    double observed_deviation = 0.0;
    // acceptable error [1% of mean and 0,5% of standard deviation]
    double max_mean_error = 0.01 * generated_mean;
    double max_deviation_error = 0.005 * generated_deviation;
    int rounds = 100000;

    double[] results = new double[rounds];

    for (int i = 0; i < rounds; i++) {
      results[i] = randomizer.nextGaussian(generated_mean, generated_deviation);
      observed_mean += results[i];
    }
    observed_mean /= (double) rounds;

    for (int i = 0; i < rounds; i++) {
      observed_deviation += Math.pow(results[i] - observed_mean, 2.0);
    }
    observed_deviation = Math.sqrt(observed_deviation / (double) rounds);

    if (Math.abs(observed_mean - generated_mean) > max_mean_error) {
      fail("Mean error is beyond acceptable range");
    }

    if (Math.abs(observed_deviation - generated_deviation) 
        > max_deviation_error) {
      fail("Standard deviation error is beyond acceptable range");
    }
  }
}
