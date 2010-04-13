package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.utils;

/**
 * @author Jarek Fuks, Zbigniew Nazimek, Marcin Golebiowski
 */

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

public class EvBinary {

  private double[] cache = null;

  private int max;


  public EvBinary() {

  }


  public void precompute(int max) {

    cache = new double[max];
    cache[0] = Math.log(1);

    for (int i = 1; i < max; i++) {
      cache[i] = cache[i - 1] + Math.log(i);
    }

    this.max = max;
  }


  /**
   * @param population
   * @param values
   * @param indexes
   * @return
   */
  public static int numberOf(EvBinaryVectorIndividual[] population,
      int[] values, int[] indexes) {
    int ret = 0;
    /*
     * @@@@@@NOTE@@@@@@ DONT PUT ANY SUGAR HERE BOA spends most his time in this
     * function and it must use as less CPU as possible!!!
     */
    if (values.length == 0)
      return population.length;

    // creating individual table for better performance

    int pop_size = population.length;

    boolean compare = true; // for comparing
    int j = 0; // for comparing
    int indexes_length = indexes.length; // for comparing

    int k = 0;
    int i = 0;

    for (i = 0; i < pop_size; i++) {

      // public static boolean compare - for better perfomance
      j = 0;
      compare = true;
      for (k = 0; k < indexes_length; k++) {
        if (population[i].getGene(indexes[k]) != values[j++]) {
          compare = false;
          break;
        }
      }
      // end of public static boolean compare

      if (compare)
        ret++;
    }

    return ret;
  }


  public static boolean[] intToBools(int bools, int size) {
    boolean[] ret = new boolean[size];

    for (int i = 0; i < size; i++) {
      if (bools % 2 != 0) {
        ret[i] = true;
      } else {
        ret[i] = false;
      }
      bools /= 2;
    }
    return ret;
  }


  public static int pow2(int n) {
    return new int[] {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
        8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
        4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
        536870912, 1073741824}[n];
  }


  public static double factorial(int n) {
    double ret = 1.0;
    for (int i = 2; i <= n; i++) {
      ret *= i;
    }
    return ret;
  }


  public double sumLogFromMap(int n) {
    if (n > max) {
      return sumLog(n);
    } else {
      return cache[n - 1];
    }
  }


  public static double sumLog(int n) {
    double ret = 0.0;
    for (int i = 1; i <= n; i++)
      ret += Math.log(i);
    return ret;
  }


  public static EvTriple numberOf(EvBinaryVectorIndividual[] array,
      int[] parents_values, int[] parents_indexes, int i) {

    int array_size = array.length;
    int parents_index_len = parents_indexes.length;

    int m_parents = 0;
    int m_false = 0;
    int m_true = 0;
    boolean equal;
    int k;

    for (k = 0; k < array_size; k++) {

      equal = true;

      for (int l = 0; l < parents_index_len; l++) {
        if (array[k].getGene(parents_indexes[l]) != parents_values[l]) {
          equal = false;
          break;
        }
      }

      if (equal) {
        m_parents++;

        if (array[k].getGene(i) == 1) {
          m_true++;
        } else {
          m_false++;
        }
      }

    }
    return new EvTriple(m_true, m_false, m_parents);
  }


  public static EvTriple[] numberOf(EvBinaryVectorIndividual[] population,
      int[] parents_indexes, int i) {
    // init

    int len = parents_indexes.length;
    int len_pow = EvBinary.pow2(len);
    EvTriple[] triples = new EvTriple[len_pow];
    int[][] parents_values = new int[len_pow][len];

    for (int k = 0; k < len_pow; k++) {
      triples[k] = new EvTriple(0, 0, 0);
    }

    for (int k = 0; k < len_pow; k++) {
      EvBinary.intToBools(k, parents_values[k], len);
    }

    // iterate over individuals

    int array_size = population.length;

    boolean equal = false;
    int k;
    int max = EvBinary.pow2(len);

    for (k = 0; k < array_size; k++) {
      for (int s = 0; s < max; s++) {
        equal = true;

        for (int l = 0; l < len; l++) {
          if (population[k].getGene(parents_indexes[l]) != parents_values[s][l]) {
            equal = false;
            break;
          }
        }

        if (equal) {

          triples[s].z = triples[s].z + 1;
          if (population[k].getGene(i) == 1) {
            triples[s].x = triples[s].x + 1;
          } else {
            triples[s].y = triples[s].y + 1;
          }
        }

      }
    }

    return triples;
  }


  public static void intToBools(int k, int[] bs, int len) {
    for (int i = 0; i < len; i++) {
      if (k % 2 != 0) {
        bs[i] = 1;
      } else {
        bs[i] = 0;
      }
      k /= 2;
    }

  }


  public void precomputeBoolsTable(int k) {

  }
}
