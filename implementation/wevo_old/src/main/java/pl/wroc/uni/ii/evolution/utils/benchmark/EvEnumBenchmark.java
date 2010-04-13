package pl.wroc.uni.ii.evolution.utils.benchmark;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Enumeration containing all the benchmarks run.
 * 
 * @author Krzysztof Sroka
 */
public enum EvEnumBenchmark {

  /**
   * Native sorting benchmark. Uses <code>Collections</code> class to sort the
   * <code>ArrayList</code> of <code>Integer</code>s, initially sorted
   * backwards.
   * 
   * @see java.util.Collections#sort(java.util.List)
   */
  NATIVE_SORT("Native sorting") {

    public double benchmark() {
      return aux_benchmark(40, 1000000, 850);
    }


    private double aux_benchmark(int rounds, int size, int time) {

      double mean = 0.0;

      ArrayList<Integer> intList = new ArrayList<Integer>(size);

      for (int i = 0; i < rounds; i++) {
        for (int j = size; j > 0; j--) {
          intList.add(new Integer(j));
        }

        long start_time = System.currentTimeMillis();

        Collections.sort(intList);

        mean += (double) (System.currentTimeMillis() - start_time);

        intList.clear();
      }

      System.out.println(getName() + " " + mean / rounds);

      return time * rounds / mean;
    }
  },

  /**
   * Sieve of Eratosthenes benchmark.
   */
  SIEVE("Sieve") {

    public double benchmark() {
      return aux_benchmark(75, 1000000, 150);
    }


    private double aux_benchmark(int rounds, int size, int time) {

      double mean = 0.0;

      ArrayList<Integer> intList = new ArrayList<Integer>(size);

      for (int i = 0; i < rounds; i++) {

        for (int j = 0; j < size; j++) {
          intList.add(new Integer(j));
        }

        long start_time = System.currentTimeMillis();

        for (int k = 2; k < Math.sqrt(size); k++) {
          if (intList.get(k) != null) {
            for (int j = k * 2; j < size; j += k) {
              intList.set(j, null);
            }
          }
        }

        mean += (double) (System.currentTimeMillis() - start_time);

        intList.clear();
      }

      System.out.println(getName() + " " + mean / rounds);

      return time * rounds / mean;
    }
  },

  /**
   * <i>Shootout</i> benchmark, realized by the <code>Shootout</code> class.
   * 
   * @see Shootout
   */
  SHOOTOUT("Shootout") {

    public double benchmark() {
      return aux_benchmark(150, 100, 50);
    }


    private double aux_benchmark(int rounds, int size, int time) {

      double mean = 0.0;

      for (int i = 0; i < rounds; i++) {
        mean += EvShootout.battle(size);
      }

      System.out.println(getName() + " " + mean / rounds);

      return time * rounds / mean;
    }
  },

  /**
   * Grabage collector benchmark. Creates series of <code>Integer</code>
   * objects and forces garbage collector to wipe whem out (by reaching upper
   * memory bound).
   */
  GARBAGE("Garbage collecting") {

    public double benchmark() {
      return aux_benchmark(25, 300);
    }


    private double aux_benchmark(int rounds, int time) {

      long size = Runtime.getRuntime().maxMemory() / 4;
      double mean = 0.0;

      @SuppressWarnings("unused")
      Integer my_integer = null;

      for (int k = 0; k < rounds; k++) {

        long start_time = System.currentTimeMillis();

        for (int i = 0; i < size; i++)
          my_integer = new Integer(i * k);

        mean += (System.currentTimeMillis() - start_time);
      }

      System.out.println(getName() + " " + mean / rounds);

      return time * rounds / mean;
    }

  };

  private String name;


  // for the sake of java constructor calls
  private EvEnumBenchmark() {
  }


  // creates enum with defined id
  private EvEnumBenchmark(String str) {
    if (str == null) {
      throw new NullPointerException();
    }

    name = str;
  }


  /**
   * @return Benchmark's name
   */
  public String getName() {
    return name;
  }


  abstract public double benchmark();
}
