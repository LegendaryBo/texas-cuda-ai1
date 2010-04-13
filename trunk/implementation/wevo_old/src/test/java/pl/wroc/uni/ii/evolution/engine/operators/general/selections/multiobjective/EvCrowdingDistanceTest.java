package pl.wroc.uni.ii.evolution.engine.operators.general.selections.multiobjective;

import static org.junit.Assert.assertTrue;
import org.junit.Test;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvBinaryPattern;

/** @author Adam Palka */
public class EvCrowdingDistanceTest {

  /**
   * Test for two objective functions and EvBinaryVectorIndividual.
   */
  @Test
  public void testForBinaryPatternTwoObjectiveFunctions() {
    EvPopulation<EvBinaryVectorIndividual> population =
      new EvPopulation<EvBinaryVectorIndividual>();    
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 0, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 1, 0}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0, 0}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0, 0}));
    for (EvBinaryVectorIndividual b : population) {
      b.addObjectiveFunction(new EvBinaryPattern(new int[] {0, 0, 1, 1}));
      b.addObjectiveFunction(new EvBinaryPattern(new int[] {1, 1, 1, 1}));
    }
    EvCrowdingDistance<EvBinaryVectorIndividual> distance =
      new EvCrowdingDistance<EvBinaryVectorIndividual>(population);
    
    assertTrue("Wrong answer for individual 0",
        distance.getCrowdingDistance(0) == Double.MAX_VALUE);
    assertTrue("Wrong answer for individual 1",
        distance.getCrowdingDistance(1) == Double.MAX_VALUE);
    assertTrue("Wrong answer for individual 2",
        distance.getCrowdingDistance(2) == 2.0);
    assertTrue("Wrong answer for individual 3",
        distance.getCrowdingDistance(3) == Double.MAX_VALUE);
    assertTrue("Wrong answer for individual 4",
        distance.getCrowdingDistance(4) == Double.MAX_VALUE);
    assertTrue("Wrong answer for individual 5",
        distance.getCrowdingDistance(5) == 4.0);
  }
}