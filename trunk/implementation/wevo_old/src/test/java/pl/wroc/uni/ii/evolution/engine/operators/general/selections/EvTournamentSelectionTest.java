package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import junit.framework.TestCase;

public class EvTournamentSelectionTest extends TestCase {

  public void testApply() {
    EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>(5);

    population.add(new EvBinaryVectorIndividual(new int[] {0, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0}));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0}));
    for(EvBinaryVectorIndividual b: population) {
      b.setObjectiveFunction(new EvOneMax());
    }

    EvTournamentSelection<EvBinaryVectorIndividual> sel = new EvTournamentSelection<EvBinaryVectorIndividual>(4, 3);
    EvPopulation<EvBinaryVectorIndividual> population2 = sel.apply(population);
    
    assertEquals(population.size(), population2.size());
      
    
    
  }

}
