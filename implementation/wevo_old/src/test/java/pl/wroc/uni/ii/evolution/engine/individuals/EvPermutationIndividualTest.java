package pl.wroc.uni.ii.evolution.engine.individuals;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;

/**
 * A simple test class of EvPermutationIndividual. 
 * 
 * @author Karol "Asgaroth" Stosiek (karol.stosiek@gmail.com)
 *
 */
public class EvPermutationIndividualTest extends TestCase {

	public void testClone() {
		EvPermutationIndividual permutation = 
			new EvPermutationIndividual(new int [] {1,2,3,4,5,6,7,8,9,0});
		
		EvPermutationIndividual clone = permutation.clone();
		
		assertTrue(permutation.equals(clone));
	}
	
	public void testIndexOf() {
		EvPermutationIndividual permutation = 
			new EvPermutationIndividual(new int [] {0,1,2,3,4,5,6,7,8,9});
		
		for (int i = 0; i < permutation.getChromosome().length; i++) {
			assertTrue(permutation.indexOf(i) == i);
		}
	}
}
