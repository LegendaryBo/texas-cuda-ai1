package pl.wroc.uni.ii.evolution.objectivefunctions.messy;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyMaxSum;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyObjectiveFunction;

public class EvSimplifiedMessyMaxSumTest extends TestCase {

  // testing individual without values
  public void testEmptyIndividual() {
    EvSimplifiedMessyIndividual individual = new EvSimplifiedMessyIndividual(4);
    EvSimplifiedMessyObjectiveFunction max_sum = new EvSimplifiedMessyObjectiveFunction(2, new EvSimplifiedMessyMaxSum(), 100);
    assertTrue(max_sum.evaluate(individual) != 0.0);
  }

  // testing individual with zero genes values only
  public void testZeroIndividual() {
    EvSimplifiedMessyIndividual individual = new EvSimplifiedMessyIndividual(4);
    EvSimplifiedMessyObjectiveFunction max_sum = new EvSimplifiedMessyObjectiveFunction(2, new EvSimplifiedMessyMaxSum(), 10);
    individual.setObjectiveFunction(max_sum);
    for (int i = 0; i < individual.getLength(); i++) {
      individual.addGeneValue(i, 0);
    }
    assertEquals(max_sum.evaluate(individual), 0.0);
  }
}
