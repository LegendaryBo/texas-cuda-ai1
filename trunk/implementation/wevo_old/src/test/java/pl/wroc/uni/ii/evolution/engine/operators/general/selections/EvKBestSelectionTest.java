package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

public class EvKBestSelectionTest extends TestCase {
  public void testKBestSelectionTest() throws Exception {
    EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>(5);
    
    population.add(new EvBinaryVectorIndividual(new int[] {0,1,1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0,0,0}));
    population.add(new EvBinaryVectorIndividual(new int[] {0,0,1}));
    population.add(new EvBinaryVectorIndividual(new int[] {1,1,1}));
    for(EvBinaryVectorIndividual b: population) {
      b.setObjectiveFunction(new EvOneMax());
    }
    
    EvOperator<EvBinaryVectorIndividual> o = new EvKBestSelection<EvBinaryVectorIndividual>(2);
    population = o.apply(population);
    assertEquals(2, population.size());
    
    assertEquals(3.0,population.get(0).getObjectiveFunctionValue());
    assertEquals(2.0,population.get(1).getObjectiveFunctionValue());
  }
}
