package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.experimental;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import junit.framework.TestCase;

/**
 * Class testing EvPermutationKRotationMutation
 * 
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 * @author Karol Asgaroth Stosiek (szymek.fogiel@gmail.com)
 */
public class EvPermutationKRotationMutationTest extends TestCase {	
	private int chromosome_length;
	private int number_of_genes_mutated;	
	private boolean exception_thrown;
	private EvPermutationIndividual individual;		
	private EvPermutationKRotationMutation mutation;
	
	
	public void setUp() {		
		number_of_genes_mutated = 4;
		exception_thrown = false;
		
		individual = new EvPermutationIndividual
			(new int[] {0,1,2,3,4,5,6,7,8});
		
		chromosome_length = individual.getChromosome().length;
	}
	
	
	public void testSuccessfulMutation() {
		assertEquals(individual.getChromosome().length, chromosome_length);		
		
		mutation = new EvPermutationKRotationMutation
			(1.0, number_of_genes_mutated);
		
		EvPermutationIndividual mutated_individual =
			mutation.mutate(individual);				
		
		int[] original_chromosome = individual.getChromosome();
		int[] mutated_chromosome = mutated_individual.getChromosome();
		
		/*number of positions on which original and mutated chromosome differ*/ 
		int different = 0;
		
		/*count positions on which chromosomes differ*/
		for (int i = 0; i < chromosome_length; i++) {
			if(original_chromosome[i] != mutated_chromosome[i]) {
				different++;
			}
		}
		
		/*different should equal number_of_genes_mutated*/
		assertEquals(different, number_of_genes_mutated);
	}
	
	
	public void testTooBigNumberOfGenesMutatedParameter() {						
		try {
			mutation = new EvPermutationKRotationMutation(0.2, 10);
			mutation.mutate(individual);
		} catch (IllegalArgumentException e) {
			exception_thrown = true;
		}
			
		assertTrue(exception_thrown);		
	}	
}
