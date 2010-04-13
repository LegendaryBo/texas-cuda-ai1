package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.EvNaturalNumberVectorChangeMutation;

public class EvMutationForNaturalNumbersTest extends TestCase {

  public void testApply() {
    EvNaturalNumberVectorIndividual ind = new EvNaturalNumberVectorIndividual(new int[] {1});
    EvNaturalNumberVectorIndividual ind2;
    //System.out.println(ind.toString());
    EvNaturalNumberVectorChangeMutation operator = new EvNaturalNumberVectorChangeMutation(0.5, 2);
    ind2 = operator.apply(new EvPopulation<EvNaturalNumberVectorIndividual>
    (new EvNaturalNumberVectorIndividual[] { ind })).get(0);
    boolean same = false;
    
    if (ind2.getNumberAtPosition(0) == 0) {
      same = true;
    }
    
    
    if (ind2.getNumberAtPosition(0) == 1) {
      same = true;
    }
    
    if (ind2.getNumberAtPosition(0) == 2) {
      same = true;
    }  
    
    if (ind2.getNumberAtPosition(0) == 3) {
      same = true;
    }  
    //System.out.println(ind2.toString());
    assertTrue(same);
  }
  

}
