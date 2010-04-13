package pl.wroc.uni.ii.evolution.sampleimplementation.students.sabinafabiszewska;

/**
 * @author Sabina Fabiszewska
 */
public final class Zad2 {

  // off MagicNumber

  /**
   * 
   */
  private static int chromosomeLength = 10;

  /**
   * 
   */
  private static int population_size = 100;

  /**
   * 
   */
  private static int max_iteration = 100;

  /**
   * 
   */
  private static double mutation_probability = 0.01;


  // on MagicNumber

  /**
   * 
   */
  private Zad2() {
  }


  /**
   * @param args (none)
   */
  public static void main(final String[] args) {
    EvMyAlgorithm alg =
        new EvMyAlgorithm(chromosomeLength, population_size, max_iteration,
            mutation_probability);
    alg.init();
    alg.run();
    System.out.println("MyAlgorithm + MyObjectiveFunction");
    System.out.println("best result: " + alg.getBestResult());
  }

}
