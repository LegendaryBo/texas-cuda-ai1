/**
 * 
 */
package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;
import junit.framework.TestCase;

/**
 * Class testing Sequence insertion mutation operator.
 * 
 * @author Karol Asgaroth Stosiek (karol.stosiek@gmail.com)
 * @author Szymon Fogiel (szymek.fogiel@gmail.com)
 */
public class EvPermutationSequenceInsertionMutationTest extends TestCase {
	
	private EvPermutationIndividual original_individual;
	private EvPermutationIndividual mutated_individual;
	private EvPermutationSequenceInsertionMutation mutation;
		
	/**
	 *  We test the mutation operator through a set of probabilities;
	 *  each test consists of a number of testSingleMutation
	 *  tests.
	 */
	public void testMutationByProbability() {
		double tested_probability = 0.0;  // probability to test
		double probability_step = 0.05;  // probability shift for each test
		
		int single_tests_per_probability = 10;
		
		for (tested_probability = 0.0; 
			tested_probability <= 1.0; 
			tested_probability += probability_step) {
			
			for (int i = 0; i < single_tests_per_probability; i++) {
				testSingleMutation(tested_probability);
			}
		}		
	}
	
	/**
	 * Test description:
	 * 
	 * In first step, we create a "window". This 
	 * window "crops" both chromosomes into two 
	 * parts. Each part (one part for one chromosome)
	 * consists of two sequences x, y. If the tested
	 * operator works, the original chromosome (cropped)
	 * consists of x, y, and the mutated chromosome
	 * consists of y, x. For example:
	 * 
	 * Original chromosome: (0,1,2,3,4,5,6,7)
	 * Mutated chromosome:  (0,1,4,5,6,2,3,7)
	 * 
	 * The window crops first chromosome to (2,3,4,5,6)
	 * and second chromosome to (4,5,6,2,3), where
	 * x = (2,3)
	 * y = (4,5,6)
	 *
	 * @param tested_probability probability of mutation.
	 */
	private void testSingleMutation(double tested_probability) {
		
		int chromosome_length = 10;
		
		original_individual = new EvPermutationIndividual(
				EvRandomizer.INSTANCE.nextPermutation(chromosome_length));
		
		mutation = 
			new EvPermutationSequenceInsertionMutation(tested_probability);
		
		mutated_individual = mutation.mutate(original_individual);
		
		int[] window_bounds = findWindowBounds();
		
		/* if the window is not empty, i.e.
		 * if the left bound is less or equal the right bound */
		if (window_bounds[0] <= window_bounds[1] ) {
			checkChromosomesConsistency(window_bounds[0], window_bounds[1]);
		}
		
		checkIfPermutationIsValid(mutated_individual);
	}
	
	private void checkChromosomesConsistency(
			int left_window_bound, int right_window_bound) {
		
		/* we analyze the first part:
		 * -> we find it's beginning in the resulting individual
		 * -> we check, if very value from the first part 
		 *    in the original individual match value on
		 *    appropriate position in mutated individual 
		 *    
		 * i iterates on the original individual;
		 * j iterates on the mutated individual. */
		int i = left_window_bound;
		int j = mutated_individual.indexOf(
				original_individual.getGeneValue(i));
		
		/* we must check, if we did not run out of the window */
		assertTrue(j <= right_window_bound);
		assertTrue(j >= left_window_bound);
		
		/* we check, if values match */
		while (j <= right_window_bound) {
			assertEquals(original_individual.getGeneValue(i),
					mutated_individual.getGeneValue(j));
			
			i++;
			j++;
		}
		
		/* we analyze the second part the same way we did 
		 * with the first part.
		 *    
		 * k iterates on the original individual;
		 * l iterates on the mutated individual. */
		int k = i;
		int l = mutated_individual.indexOf(
				original_individual.getGeneValue(k));
		
		/* we must check, if we did not run out of the window */
		assertTrue(l <= right_window_bound);
		assertTrue(l >= left_window_bound);
		
		/* we check, if values match */
		while (k <= right_window_bound) {
			assertEquals(original_individual.getGeneValue(k),
					mutated_individual.getGeneValue(l));
			
			k++;
			l++;
		}
	}
	
		
	private int[] findWindowBounds() {
		int chromosome_length = original_individual.getChromosome().length;
		
		/* finding the left window bound. */
		int left_window_bound = 0;	// left window bound
		while (left_window_bound < chromosome_length) {
			
			/* gene values do not match - we've 
			 * entered the window. */
			if (original_individual.getGeneValue(left_window_bound) != 
					mutated_individual.getGeneValue(left_window_bound)) {
				break;
			}
			
			left_window_bound++;
		}
		
		/* finding the right window bound. */
		int right_window_bound = chromosome_length - 1;	// right window bound
		while (right_window_bound  >= 0) {
			
			/* gene values do not match - we've 
			 * entered the window */
			if (original_individual.getGeneValue(right_window_bound ) != 
					mutated_individual.getGeneValue(right_window_bound )) {
				break;
			}
		
			right_window_bound--;
		}
		
		int[] window_bounds = new int[2];
		window_bounds[0] = left_window_bound;
		window_bounds[1] = right_window_bound;
		
		return window_bounds;
	}
	
	/**
	 * Method for checking, if a given individual
	 * has a valid permutation as a chromosome.
	 * 
	 * @param individual permutation individual to check
	 */
	private void checkIfPermutationIsValid(
			EvPermutationIndividual individual) {
		
		int chromosome_length = individual.getChromosome().length;

		/* counting the number of occurrences of each value */
		int value;
		int[] times_exist = new int[chromosome_length];
		for(int i = 0; i < chromosome_length; i++){
			value = individual.getGeneValue(i);
			times_exist[individual.indexOf(value)]++;
		}
		
		/* now it is to be checked, if each value in permutation
		 * was encountered only once */
		for (int i = 0; i < chromosome_length; i++) {
			assertEquals(times_exist[i], 1);
		}
	}
}
