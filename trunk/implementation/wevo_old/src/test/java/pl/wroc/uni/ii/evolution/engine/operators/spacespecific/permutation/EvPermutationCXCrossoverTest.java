package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.EvPermutationCXCrossover;
import junit.framework.TestCase;

/**
 *  Testing class for EvPermutationCXCrossover class. 
 *  We assume, that individuals are valid.
 *  
 *  Usual case test:
 *  
 *  Parents:
 *  Parent 1: 8 4 7 3 6 2 5 1 9 0
 *	Parent 2: 0 1 2 3 4 5 6 7 8 9
 *	
 *  should result in:
 *	Child 1:  8 1 2 3 4 5 6 7 9 0
 *	Child 2:  0 4 7 3 6 2 5 1 8 9  
 *
 * @author Karol "Asgaroth" Stosiek (karol.stosiek@gmail.com)
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvPermutationCXCrossoverTest extends TestCase {
   
    /**
     * Testing, if CX operator fills every gene position
     * in child chromosomes, basing on the fact, that every
     * negative gene value is invalid.
     */
	public void testFullyFillsTheChildChromosomes() {
		EvPermutationIndividual parent1 = 
			new EvPermutationIndividual(new int[] {1,5,4,2,3,0});
	    
	    EvPermutationIndividual parent2 = 
	    	new EvPermutationIndividual(new int[] {0,1,5,4,3,2});
		  
	    List<EvPermutationIndividual> parents = 
	    	new ArrayList<EvPermutationIndividual>();
	    
	    parents.add(parent1);
	    parents.add(parent2);
	    
	    EvPermutationCXCrossover crossover = 
	    	new EvPermutationCXCrossover();
	   
	    List<EvPermutationIndividual> result = 
	    	new ArrayList<EvPermutationIndividual>();  
	    
	    result = crossover.combine(parents);

	    EvPermutationIndividual child1 = result.get(0);
	    EvPermutationIndividual child2 = result.get(1);
	    
	    boolean filled_correctly = true;
	    for (int i = 0; i < parent1.getChromosome().length; i++) {
	    	if (child1.getGeneValue(i) < 0 || child2.getGeneValue(i) < 0) {
	    		filled_correctly = false;
	    		break;
	    	}
	    }
	   
	    assertTrue(filled_correctly);
	}
	
	/**
	 * Testing CX operator in usual case, i.e. when
	 * chromosome length is longer than 1.
	 */
	public void testPermutationsLongerThanOne() {
		EvPermutationIndividual parent1 = 
			new EvPermutationIndividual(new int[] {8,4,7,3,6,2,5,1,9,0});
    
		EvPermutationIndividual parent2 = 
			new EvPermutationIndividual(new int[] {0,1,2,3,4,5,6,7,8,9});
    
		EvPermutationIndividual objective_child1 = 
			new EvPermutationIndividual(new int[] {8,1,2,3,4,5,6,7,9,0});
    
		EvPermutationIndividual objective_child2 = 
			new EvPermutationIndividual(new int[] {0,4,7,3,6,2,5,1,8,9});
	  
		List<EvPermutationIndividual> parents = 
			new ArrayList<EvPermutationIndividual>();
    
		parents.add(parent1);
		parents.add(parent2);
    
		EvPermutationCXCrossover crossover = 
			new EvPermutationCXCrossover();
   
		List<EvPermutationIndividual> result = 
			new ArrayList<EvPermutationIndividual>();  
    
		result = crossover.combine(parents);

		EvPermutationIndividual child1 = result.get(0);
		EvPermutationIndividual child2 = result.get(1);
   
		assertTrue(child1.equals(objective_child1));
		assertTrue(child2.equals(objective_child2));
	}

	/**
	 * Testing, if CX operator works for short (of length 1)
	 * chromosomes.
	 */
	public void testPermutationsOfLengthOne() {
		EvPermutationIndividual parent1 = 
			new EvPermutationIndividual(new int[] {0});
	    
	    EvPermutationIndividual parent2 = 
	    	new EvPermutationIndividual(new int[] {0});
	    
	    EvPermutationIndividual objective_child1 = 
	    	new EvPermutationIndividual(new int[] {0});
	    
	    EvPermutationIndividual objective_child2 = 
	    	new EvPermutationIndividual(new int[] {0});
		  
	    List<EvPermutationIndividual> parents = 
	    	new ArrayList<EvPermutationIndividual>();
	    
	    parents.add(parent1);
	    parents.add(parent2);
	    
	    EvPermutationCXCrossover crossover = 
	    	new EvPermutationCXCrossover();
	   
	    List<EvPermutationIndividual> result = 
	    	new ArrayList<EvPermutationIndividual>();  
	    
	    result = crossover.combine(parents);

	    EvPermutationIndividual child1 = result.get(0);
	    EvPermutationIndividual child2 = result.get(1);
	   
	    assertTrue(child1.equals(objective_child1));
	    assertTrue(child2.equals(objective_child2));
	}
}
