package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import junit.framework.TestCase;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

public class EvPermutationInversionMutationTest extends TestCase {

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
	 * We can see, that mutated individual chromosome 
	 * consists of three parts: the first and the third are copied
	 * directly from the original, the second is the
	 * inverted sequence. 
	 * 
	 * We check the first and third part for being equal
	 * with the original chromosome, and check the second part
	 * for being inverted. 
	 *
	 * @param tested_probability probability of mutation.
	 */
	private void testSingleMutation(double tested_probability) {
		EvPermutationIndividual original_individual;
		EvPermutationIndividual mutated_individual;
		EvPermutationInversionMutation mutation;
		
		int chromosome_length = 10;
		
		original_individual = new EvPermutationIndividual(
				EvRandomizer.INSTANCE.nextPermutation(chromosome_length));
		
		mutation = new EvPermutationInversionMutation(tested_probability);
		mutated_individual = mutation.mutate(original_individual);
		
		/* we check the first part; if gene values do not match,
		 * then we entered the reverted section. */
		int i = 0;
		while (i < chromosome_length) {
			if (mutated_individual.getGeneValue(i) != 
					original_individual.getGeneValue(i)){
				break;
			}
			
			i++;				
		}
		
		/* we enter the inverted section; we have to
		 * find its end. We will move backwards, to find
		 * the first non-matching values. */
		int j = chromosome_length - 1;
		while (j >= i) {
			
			/* the values do not match - we've found the end
			 * of second section. */
			if (original_individual.getGeneValue(j) !=
					mutated_individual.getGeneValue(j)) {
				break;
			}
			
			j--;
		}
		
		/* we check if the section really is inverted */
		int t1 = i;  // iterating on the original chromosome
		int t2 = j;  // iterating on the mutated chromosome
	
		/* if there is no section changed, j index will be less 
		 * than i. */
		if (j >= i) {
			while (t1 <= j && t2 >= i){
				assertEquals(original_individual.getGeneValue(t1),
							mutated_individual.getGeneValue(t2));
				
				t1++;
				t2--;
			}
		}	
		
		/* we move to the beginning of the third section 
		 * and assure, that positions from now on equal. */
		j++;
		while (j < chromosome_length) {
			assertEquals(original_individual.getGeneValue(j),
					mutated_individual.getGeneValue(j));
			j++;
		}
		
		checkIfPermutationIsValid(mutated_individual);
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
