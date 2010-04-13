package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import junit.framework.TestCase;

public class EvRandomSelectionTest extends TestCase {

  public void testGetIndexes() {
   
    EvBinaryVectorIndividual ind1 = new EvBinaryVectorIndividual(new int[] {1, 1, 1, 1});
    EvBinaryVectorIndividual ind2 = new EvBinaryVectorIndividual(new int[] {1, 1, 0, 1});
    EvBinaryVectorIndividual ind3 = new EvBinaryVectorIndividual(new int[] {1, 1, 0, 0});
    EvBinaryVectorIndividual ind4 = new EvBinaryVectorIndividual(new int[] {0, 0, 0, 0});
    EvBinaryVectorIndividual ind5 = new EvBinaryVectorIndividual(new int[] {0, 0, 1, 0});
    
    EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>();
    population.add(ind1);
    population.add(ind2);
    population.add(ind3);
    population.add(ind4);
    population.add(ind5);
    population.setObjectiveFunction(new EvOneMax());
    
    EvRandomSelection<EvBinaryVectorIndividual> selection = new EvRandomSelection<EvBinaryVectorIndividual>(2, false);
   
    assertEquals(2, selection.getIndexes(population).size());
   
    
    for (EvBinaryVectorIndividual ind: selection.apply(population)) {
      System.out.println(ind);
    }
    
  }

}
