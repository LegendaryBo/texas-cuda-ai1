package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import junit.framework.TestCase;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvPairsTournamentSelection;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

/**
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvPairsTournamentSelectionTest extends TestCase {

  // Test of parallel tournament
  public void testApply() {
    EvPopulation<EvBinaryVectorIndividual> population =
        new EvPopulation<EvBinaryVectorIndividual>(4);

    population.add(new EvBinaryVectorIndividual(new int[] {0, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0}));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0}));
    for(EvBinaryVectorIndividual individual: population) {
      individual.setObjectiveFunction(new EvOneMax());
    }

    EvPairsTournamentSelection<EvBinaryVectorIndividual> sel =
        new EvPairsTournamentSelection<EvBinaryVectorIndividual>(3);
    EvPopulation<EvBinaryVectorIndividual> population2 = sel.apply(population);
    
    assertEquals(population2.size(), 3);
    // There must be {1, 1, 1} individual with 3.0 objective function value
    assertEquals(population2.getBestResult().getObjectiveFunctionValue(), 3.0); 
  }
  
  
  // Test of online sequenced selection
  public void testGetNextIndex() {
    EvPopulation<EvBinaryVectorIndividual> population =
        new EvPopulation<EvBinaryVectorIndividual>(5);

    population.add(new EvBinaryVectorIndividual(new int[] {0, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0}));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0}));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 0}));
    for(EvBinaryVectorIndividual individual: population) {
      individual.setObjectiveFunction(new EvOneMax());
    }

    EvPairsTournamentSelection<EvBinaryVectorIndividual> sel =
        new EvPairsTournamentSelection<EvBinaryVectorIndividual>(2);
    
    sel.init(population, 4, 0);
    for (int i = 0; i < 16; i++) {
      EvBinaryVectorIndividual individual1 = population.get(sel.getNextIndex());
      EvBinaryVectorIndividual individual2 = population.get(sel.getNextIndex());
      EvBinaryVectorIndividual individual3 = population.get(sel.getNextIndex());
      EvBinaryVectorIndividual individual4 = population.get(sel.getNextIndex());
    
      // Individuals 1, 2 and 3, 4 must be distinct
      assertFalse(individual1.equals(individual2));
      assertFalse(individual3.equals(individual4));
    }
    
    // Test of getNextParents
    List<EvBinaryVectorIndividual> list = sel.getNextParents();
    assertEquals(list.size(), 4);
    assertFalse(list.get(0).equals(list.get(1)));
    assertFalse(list.get(2).equals(list.get(3)));
    
  }

}