package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.experimental;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;
import junit.framework.TestCase;

public class EvPermutationShuffleMutationTest extends TestCase {

	/**
	 * Performs series of tests, first fixing a probability,
	 * then fixing the fraction. 
	 */
	public void testMutationByProbability()
	{
		double tested_probability = 0.0;  // probability to test
		double probability_step = 0.05;	// probability shift for each step
		
		int single_tests_per_probability = 10;
		
		for (tested_probability = 0.0; 
				tested_probability <= 1.0; 
				tested_probability += probability_step) {
	
			for (int i = 0; i < single_tests_per_probability; i++) {
				testMutationByfraction(tested_probability);
			}
		}			
	}
	
	/**
	 * Performs tests for a set of fractions; only fraction 
	 * varies; mutation probability is constant.
	 * 
	 * @param mutation_probability probability of a mutation
	 */
	private void testMutationByfraction(double mutation_probability)
	{
		double tested_fraction = 0.0;
		double fraction_step = 0.05;
		
		for (tested_fraction = 0.0;
				tested_fraction <= 1.0;
				tested_fraction += fraction_step) {
			
			testSingleMutation(mutation_probability, tested_fraction);
		}
	}
	
	/**
	 * Single mutation application test. 
	 * 
	 * @param probability probability of mutation.
	 * @param fraction fraction of genes that are to be shuffled.
	 */
	private void testSingleMutation(double probability, double fraction) {
		EvPermutationIndividual original_individual;
		EvPermutationIndividual mutated_individual;
		EvPermutationShuffleMutation mutation;
		
		int chromosome_length = 10;
		
		original_individual = new EvPermutationIndividual(
				EvRandomizer.INSTANCE.nextPermutation(chromosome_length));
		
		mutation = new EvPermutationShuffleMutation(probability, fraction);
		mutated_individual = mutation.mutate(original_individual);
		
		/* counting the values, that do not match */
		int not_matching_values = 0;
		for (int i = 0; i < chromosome_length; i++) {
			if (original_individual.getGeneValue(i) !=
					mutated_individual.getGeneValue(i)) {
				not_matching_values++;
			}
		}
		
		/* we calculate the expected number of values,
		 * that do not match */
		int expected_not_matching_values = 
			(int)Math.floor(fraction * chromosome_length);
		
		/* since shuffling can result in identity,
		 * the only thing we can assure is that 
		 * there are no more modified positions than
		 * we expect */
		assertTrue(not_matching_values <= expected_not_matching_values);
		
		/* checking permutation validity */
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
