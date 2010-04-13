package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import junit.framework.TestCase;

/**
 * Class testing if EvPermutationOPXCrossover works fine.
 * Very naive test. Maybe there should be more testing cases.
 * The good news is that class analyzed "by hand" 
 * (with analyzing each step printed out to the standard output)
 * seemed to work fine. 
 * 
 * @author Karol "Asgaroth" Stosiek (karol.stosiek@gmail.com) 
 * @author Szymek "Shnook" Fogiel (szymek.fogiel@gmail.com)
 */
public class EvPermutationOPXCrossoverTest extends TestCase {
	
	private EvPermutationIndividual parent1;
	private EvPermutationIndividual parent2;
	private ArrayList<EvPermutationIndividual> parents;
	private int chromosome_length = 9;
	
	@Override
	protected void setUp() throws Exception {
		
		chromosome_length = 9;
		
		parent1 = new EvPermutationIndividual(new int[] {0,1,2,3,4,5,6,7,8});
		parent2 = new EvPermutationIndividual(new int[] {3,0,1,7,6,5,8,2,4});
	
		assertEquals(parent1.getChromosome().length, chromosome_length);
		assertEquals(parent2.getChromosome().length, chromosome_length);
		
		parents = new ArrayList<EvPermutationIndividual>();
		
		parents.add(parent1);
		parents.add(parent2);	
	}
	
	public void testCrossoverWithDifferentFractions() {
		
		/* we check, if the operator works properly
		 * for a few fraction values. */
		double fraction = 0.0;
		for (int i = 0; i < 100; i++, fraction += 1/100) {
			testCrossoverWithFraction(fraction);
		}
	}
	
	private void testCrossoverWithFraction(double fraction_from_first_parent) {		
		
		/* we create new crossover operator with */
		EvPermutationOPXCrossover crossover = 
			new EvPermutationOPXCrossover(fraction_from_first_parent);
		
		/* These variables contain the number of genes,
		 * that that differ in child and its parents.
		 * (in parent1 and parent2, respectively) */
		int positions_not_differing_with_parent2 = 0;
		
		int tests = 10;  // number of tests to do
		
		/* Variable positions_not_changed holds the number
		 * of positions, that must exist both in child 
		 * and its second parent on the same positions. */
		double positions_not_changed = 0;   
		
		/* the only child resulting in crossing over */
		EvPermutationIndividual child;

		positions_not_differing_with_parent2 = 0;
		for (int test = 0; test < tests; test++){
			
			/* we cross over the parents */
			List<EvPermutationIndividual> children = crossover.combine(parents);
			child = children.get(0);
			
			/* counting the positions, on which child 
			 * differs with its parents */
			positions_not_differing_with_parent2 = 0;
			for (int i = 0; i < parent1.getChromosome().length; i++) {
				if (child.getGeneValue(i) == parent2.getGeneValue(i)) {
					positions_not_differing_with_parent2++;
				}
			}
			
			/* test, if the resulting permutation is valid */
			checkIfPermutationIsValid(child);
			
			/* we calculate the number of positions, 
			 * that must not change */
			positions_not_changed = 
				(int)(fraction_from_first_parent * chromosome_length);

			
			assertTrue(positions_not_differing_with_parent2 
					>= positions_not_changed);
		}
	}
	
	private void checkIfPermutationIsValid(
			EvPermutationIndividual permutation) {
		
		/* filling the existence array (to compare with) */
		int[] value_exists_in_permutation_once = new int[chromosome_length];
		for (int i = 0; i < chromosome_length; i++) {
			value_exists_in_permutation_once[i] = 1;
		}

		/* counting the number of occurrences of each value */
		int value;
		int[] value_exists_in_permutation = new int[chromosome_length];
		for(int i = 0; i < chromosome_length; i++){
			value = permutation.getGeneValue(i);
			value_exists_in_permutation[permutation.indexOf(value)]++;
		}
		
		/* now it is to be checked, if each value in permutation
		 * was encountered only once */
		for (int i = 0; i < chromosome_length; i++) {
			assertEquals(value_exists_in_permutation[i], 
					value_exists_in_permutation_once[i]);
		}

	}
}
