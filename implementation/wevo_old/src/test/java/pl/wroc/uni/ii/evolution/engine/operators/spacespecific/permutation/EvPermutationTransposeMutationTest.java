package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.EvPermutationTransposeMutation;
import pl.wroc.uni.ii.evolution.objectivefunctions.permutation.EvLessCycleNumber;
import junit.framework.TestCase;
/**
 * @author Donata Malecka, Piotr Baraniak
 */
public class EvPermutationTransposeMutationTest extends TestCase {
  public void testIfItWorks() {
    
    EvPermutationIndividual ind1, clone = null;

    ind1 = new EvPermutationIndividual(new int[]{1,2,3,0});
    
   
    ind1.setObjectiveFunction(new EvLessCycleNumber());
    
    EvPermutationTransposeMutation mutation = new EvPermutationTransposeMutation(1);
  
    clone  = ind1.clone();
    mutation.mutate(ind1);

   
    int not_same = 0;
    for (int i = 0; i < ind1.getChromosome().length; i++ ) {
      if(clone.getChromosome()[i] != ind1.getChromosome()[i]) not_same++;
    }
    assertEquals(2,not_same);
  }
}
