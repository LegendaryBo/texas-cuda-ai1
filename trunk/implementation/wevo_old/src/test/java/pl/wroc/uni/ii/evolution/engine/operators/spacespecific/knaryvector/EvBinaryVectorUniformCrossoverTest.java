package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import junit.framework.TestCase;

public class EvBinaryVectorUniformCrossoverTest extends TestCase {

  public void testCombineListOfEvBinaryVectorIndividual() {
   
    EvBinaryVectorSpace space = new EvBinaryVectorSpace(new EvOneMax(), 1000);
    
    EvBinaryVectorIndividual ind1 = space.generateIndividual();
    EvBinaryVectorIndividual ind2 = space.generateIndividual();
    EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual> crossover = 
      new EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual>();
    
    List<EvBinaryVectorIndividual> parents = new ArrayList<EvBinaryVectorIndividual>();
    parents.add(ind1);
    parents.add(ind2);
    
    int ones_count = (int) ind1.getObjectiveFunctionValue() + (int) ind2.getObjectiveFunctionValue();
   
    List<EvBinaryVectorIndividual> result = crossover.combine(parents);
    
    
    int result_count = 0;

    for (EvBinaryVectorIndividual ind: result) {
      result_count += ind.getObjectiveFunctionValue();
    }
    
    
    assertEquals(ones_count, result_count);
  }
  
  
 

}
