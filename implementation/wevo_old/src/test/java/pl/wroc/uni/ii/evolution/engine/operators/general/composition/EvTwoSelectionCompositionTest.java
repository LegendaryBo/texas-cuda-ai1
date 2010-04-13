package pl.wroc.uni.ii.evolution.engine.operators.general.composition;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import junit.framework.TestCase;

public class EvTwoSelectionCompositionTest extends TestCase {

  public void testGetIndexes() {
    
    EvKBestSelection<EvBinaryVectorIndividual> first = new EvKBestSelection<EvBinaryVectorIndividual>(3);
    EvKBestSelection<EvBinaryVectorIndividual> second = new EvKBestSelection<EvBinaryVectorIndividual>(2);
    
    
    EvTwoSelectionComposition<EvBinaryVectorIndividual> selection = new EvTwoSelectionComposition<EvBinaryVectorIndividual>(first, second);
   
     EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>(5);
    
    population.add(new EvBinaryVectorIndividual(new int[] {0, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 1}));
    for(EvBinaryVectorIndividual b: population) {
      b.setObjectiveFunction(new EvOneMax());
    }
    
    for (int i : selection.getIndexes(population)) {
      System.out.println(i);
    }
    
    
  }

}
