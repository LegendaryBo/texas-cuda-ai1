package pl.wroc.uni.ii.evolution.solutionspaces;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.permutation.EvLessCycleNumber;
import pl.wroc.uni.ii.evolution.solutionspaces.EvPermutationSpace;
import junit.framework.TestCase;
/**
 * EvPermutationSpaceTest
 *
 * @author Donata Malecka, Piotr Baraniak
 */
public class EvPermutationSpaceTest extends TestCase {
  
  public void testRandomPermutation()
  {
  
    EvPermutationSpace permSpace = new EvPermutationSpace(10);
    EvLessCycleNumber function = new EvLessCycleNumber();
    permSpace.setObjectiveFuntion(function);
    EvPermutationIndividual ind = (EvPermutationIndividual)permSpace.generateIndividual();
    assertNotNull(ind.getObjectiveFunction());
    assertNotNull(ind.getObjectiveFunctionValue());
    int[] tab = ind.getChromosome();
    int k=0;
    System.out.println(ind.toString()); 
    for (int i = 0; i<10; i++)
    {
      k = 0; 
      for (int j = 0; j<10; j++)
      {
        if (tab[j]==i)
          k++;
      }
      assertEquals(1,k);
    }


  }
  
}
