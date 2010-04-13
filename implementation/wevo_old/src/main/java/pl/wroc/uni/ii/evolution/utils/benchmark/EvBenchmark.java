package pl.wroc.uni.ii.evolution.utils.benchmark;

import java.util.Date;

/**
 * Main benchmark utility class, provides static method for running benchmark,
 * as well as main method to run benchmark as a separate application.
 * 
 * @author Krzysztof Sroka
 */
public class EvBenchmark {

  /**
   * Static method for running benchmark tests from EnumBenchmark enumeration.
   * 
   * @return int value, higher for faster systems.
   */
  public static int runBenchmark() {

    double benchmarks_sum = 0.0;

    for (EvEnumBenchmark bmark : EvEnumBenchmark.values()) {
      benchmarks_sum += bmark.benchmark();
    }

    return Double.valueOf(
        1000 * benchmarks_sum / EvEnumBenchmark.values().length).intValue();
  }


  public static double runMasterSlaveBenchmark() {
    long starting = (new Date()).getTime();

    String blah = new String();
    for (int i = 0; i < 20000; i++) {
      blah += "a";
    }

    long ending = (new Date()).getTime();
    return 1 / (double) (ending - starting);
  }


  /**
   * Method for running benchmark separately. Prints out the benchmark
   * information.
   * 
   * @param args program arguments (totally ignored :-) )
   */
  public static void main(String[] args) {
    System.out.println("Benchmark: " + runMasterSlaveBenchmark());
  }
}
