package pl.wroc.uni.ii.evolution.sampleimplementation.students.sabinafabiszewska;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvECGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * @author Sabina Fabiszewska
 */
public final class Zad1 {

  // off MagicNumber

  /**
   * 
   */
  private static int chromosome_length = 10;

  /**
   * 
   */
  private static int population_size = 100;

  /**
   * 
   */
  private static int tournament_size = 100;

  /**
   * 
   */
  private static int number_of_winners = 10;

  /**
   * 
   */
  private static int number_of_iteration = 100;


  // on MagicNumber

  /**
   * 
   */
  private Zad1() {
  }


  /**
   * @param args (none)
   */
  public static void main(final String[] args) {

    EvKDeceptiveOneMax function = new EvKDeceptiveOneMax(chromosome_length);
    EvBinaryVectorSpace space =
        new EvBinaryVectorSpace(function, chromosome_length);
    EvECGA algorithm =
        new EvECGA(true, population_size, tournament_size, number_of_winners);

    algorithm
        .setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
            number_of_iteration));
    algorithm.setSolutionSpace(space);
    algorithm.setObjectiveFunction(function);

    algorithm.init();
    algorithm.run();

    System.out.println("deceptive OneMax, ECGA");
    System.out.println("best result: " + algorithm.getBestResult());
  }
}