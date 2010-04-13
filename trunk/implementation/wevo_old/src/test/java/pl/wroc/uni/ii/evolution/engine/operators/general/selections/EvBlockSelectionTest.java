package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvBlockSelection;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import junit.framework.TestCase;

public class EvBlockSelectionTest extends TestCase {

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
    
    EvBlockSelection<EvBinaryVectorIndividual> selection = new EvBlockSelection<EvBinaryVectorIndividual>(2);
    
    List<Integer> result = selection.getIndexes(population);
    
    assertEquals(5, result.size());
    
    int[] pattern = new int[] {0, 1, 2, 1, 0};
    for (int i =0 ; i < result.size(); i++) {
      assertTrue(pattern[i] == result.get(i));
    }
    
    
  }

}
