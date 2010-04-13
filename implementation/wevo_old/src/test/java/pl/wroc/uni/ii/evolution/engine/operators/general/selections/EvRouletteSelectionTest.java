package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRouletteSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvIndividualFitness;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;

public class EvRouletteSelectionTest extends TestCase {

  public void testApply() {
    EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>(5);
    
    population.add(new EvBinaryVectorIndividual(new int[] {0, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0 }));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0 }));
    for(EvBinaryVectorIndividual b: population) {
      b.setObjectiveFunction(new EvOneMax());
    }
    
    EvIndividualFitness<EvBinaryVectorIndividual> fit = new EvIndividualFitness<EvBinaryVectorIndividual>();
    EvOperator<EvBinaryVectorIndividual> o = new EvRouletteSelection<EvBinaryVectorIndividual>(fit, 2);
    assertEquals(o.apply(o.apply(population)).size(), 2);
  }
  
  public void testRouletteWheel() throws Exception {
    EvPopulation<EvStunt> pop = EvStunt.pop(5.0,4.0,1.0);
    EvRouletteSelection.RouletteWheel<EvStunt> wheel = 
      new EvRouletteSelection.RouletteWheel<EvStunt>(pop, new EvIndividualFitness<EvStunt>());
    
    assertEquals(0, wheel.getIndivIndexAtPosition(0.1));
    assertEquals(2, wheel.getIndivIndexAtPosition(0.99));
  }

}
