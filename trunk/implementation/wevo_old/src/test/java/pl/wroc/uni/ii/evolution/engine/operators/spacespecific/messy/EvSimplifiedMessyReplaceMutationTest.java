package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvSimplifiedMessyReplaceGeneMutation;


public class EvSimplifiedMessyReplaceMutationTest extends TestCase {

  public void testZeroProbabilityApply() {
    EvSimplifiedMessyReplaceGeneMutation mutation_operator =
      new EvSimplifiedMessyReplaceGeneMutation(0.0, 5);

    EvPopulation<EvSimplifiedMessyIndividual> population = new EvPopulation<EvSimplifiedMessyIndividual>();
    
    population.add(new EvSimplifiedMessyIndividual(30));
    population.add(new EvSimplifiedMessyIndividual(10));
    population.add(new EvSimplifiedMessyIndividual(6));
    population.add(new EvSimplifiedMessyIndividual(10));
    population.add(new EvSimplifiedMessyIndividual(6));
    population.add(new EvSimplifiedMessyIndividual(2));
    
    population = new EvPopulation<EvSimplifiedMessyIndividual>(mutation_operator.apply(population));

    
    for (EvSimplifiedMessyIndividual ind: population) {
      for (int i = 0; i < ind.getLength(); i++) {
        assertTrue(ind.getGeneValues(i).size() == 0);
      }
    }
    
  }
  
  public void testOneProbabilityApply() {
    EvSimplifiedMessyReplaceGeneMutation mutation_operator =
      new EvSimplifiedMessyReplaceGeneMutation(1.0, 0);

    EvPopulation<EvSimplifiedMessyIndividual> population = new EvPopulation<EvSimplifiedMessyIndividual>();
    
    population.add(new EvSimplifiedMessyIndividual(30));
    population.add(new EvSimplifiedMessyIndividual(10));
    population.add(new EvSimplifiedMessyIndividual(6));
    population.add(new EvSimplifiedMessyIndividual(10));
    population.add(new EvSimplifiedMessyIndividual(6));
    population.add(new EvSimplifiedMessyIndividual(2));
    
    population = new EvPopulation<EvSimplifiedMessyIndividual>(mutation_operator.apply(population));

    
    for (EvSimplifiedMessyIndividual ind: population) {
      for (int i = 0; i < ind.getLength(); i++) {
        assertTrue(ind.getGeneValues(i).size() != 0);
        assertEquals(0, ind.getGeneValue(i));
      }
    }
    
  }

}
