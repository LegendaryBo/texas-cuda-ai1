package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.utils;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.utils.EvBinary;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.utils.EvTriple;
import junit.framework.TestCase;

public class EvBinaryTest extends TestCase {

  public void testNumberOfEvBinaryVectorIndividualArrayBooleanArrayIntArrayInt() {
   
    EvPopulation<EvBinaryVectorIndividual> pop = new EvPopulation<EvBinaryVectorIndividual>();
    
    pop.add(new EvBinaryVectorIndividual(new int[] {1, 1, 0, 1}));
    pop.add(new EvBinaryVectorIndividual(new int[] {1, 0, 0, 1}));
    pop.add(new EvBinaryVectorIndividual(new int[] {1, 1, 1, 1}));
    pop.add(new EvBinaryVectorIndividual(new int[] {1, 0, 1, 1}));
    
   
    
    
    EvTriple trip = EvBinary.numberOf(pop.toArray(new EvBinaryVectorIndividual[pop.size()]), 
        new int[] {1}, new int[] {1}, 0);
    
    
    assertEquals(trip.x, 2);
    assertEquals(trip.y, 0);
    assertEquals(trip.z, 2); 
   
    
    
  }

}
