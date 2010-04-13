package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvOnlyFromSpaceSelection;
import pl.wroc.uni.ii.evolution.solutionspaces.EvNaturalNumberVectorSpace;

public class EvOnlyFromSpaceSelectionTest extends TestCase {

  public void testApply() {
    EvPopulation<EvNaturalNumberVectorIndividual> pop = new EvPopulation<EvNaturalNumberVectorIndividual>();
    
    pop.add(new EvNaturalNumberVectorIndividual(new int[] {1, 2, 3}));
    pop.add(new EvNaturalNumberVectorIndividual(new int[] {2, 3}));
    pop.add(new EvNaturalNumberVectorIndividual(new int[] {5, 2}));
    
    EvNaturalNumberVectorSpace space = new EvNaturalNumberVectorSpace(null, 3);
    
    EvOnlyFromSpaceSelection<EvNaturalNumberVectorIndividual> filtr = new EvOnlyFromSpaceSelection<EvNaturalNumberVectorIndividual>(space);
    assertEquals(1, filtr.apply(pop).size());
    
    EvNaturalNumberVectorSpace space2 = new EvNaturalNumberVectorSpace(null, 2);
    EvOnlyFromSpaceSelection<EvNaturalNumberVectorIndividual> filtr2 = new EvOnlyFromSpaceSelection<EvNaturalNumberVectorIndividual>(space2);
    assertEquals(2, filtr2.apply(pop).size());
    
  }

}
