package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.EvPermutationInsertMutation;
import pl.wroc.uni.ii.evolution.objectivefunctions.permutation.EvLessCycleNumber;
import junit.framework.TestCase;
/**
 * @author Donata Malecka, Piotr Baraniak
 */
public class EvPermutationInsertMutationTest extends TestCase {
  public void testIfItWorks() {
    int d = 10000;
    
    int[] chromosome = new int[d];
    
    
    for (int i = 0; i < d; i++) {
      chromosome[i] = i;
    }
    
    EvPermutationIndividual ind = new EvPermutationIndividual(chromosome);
    ind.setObjectiveFunction(new EvLessCycleNumber());

    
    EvPermutationInsertMutation mutation = new EvPermutationInsertMutation(0);
  
    EvPermutationIndividual cloned = ind.clone();
    
    
    mutation.mutate(cloned);
    assertEquals(ind, cloned); 

    
    mutation = new EvPermutationInsertMutation(1);
    cloned = ind.clone();
    mutation.mutate(cloned);

    int i;
    int[] n_chromosome = cloned.getChromosome();
    
    for (i = 0; i < d && chromosome[i] == n_chromosome[i] ; i++) ;
    assertTrue("There was no mutation, when probability parameter is 1",i < d);
    int a = i;
    
    for (; i < d && chromosome[i] != n_chromosome[i]; i++) ;
    
    int b;
    
    if (i >= d) { 
      b = d - 1;
    }
    else {
      b = i - 1;
    }
    for (; i < d && chromosome[i] == n_chromosome[i] ; i++) ;
    assertTrue("There is second block of changed elements.",i >= d);

    if (n_chromosome[a] == chromosome[a+1]) {

      assertTrue(n_chromosome[b] == chromosome[a]);
      for(int j = a + 1 ; j < b; j++) assertTrue(n_chromosome[j] == chromosome[j+1]);
      
    } else {

      assertTrue(n_chromosome[a] == chromosome[b]);
      for(int j = a + 1 ; j <= b; j++) assertTrue(n_chromosome[j] == chromosome[j-1]);
    }
  }
}
