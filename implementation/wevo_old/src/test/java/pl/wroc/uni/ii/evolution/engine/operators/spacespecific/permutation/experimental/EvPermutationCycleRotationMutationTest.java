package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.experimental;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import junit.framework.TestCase;

/**
 * Class testing EvPermutationKRotationMutation
 * 
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 * @author Karol Asgaroth Stosiek (szymek.fogiel@gmail.com)
 */
public class EvPermutationCycleRotationMutationTest extends TestCase {	
	private int chromosome_length;
	
	/*number of genes that should change after mutation*/
	private int number_of_genes_mutated;
	private EvPermutationIndividual individual;
	private EvPermutationCycleRotationMutation mutation;
	
	
	public void setUp() {		
		/*there's only one cycle*/
		individual = new EvPermutationIndividual
			(new int[] {2,3,4,5,6,7,8,0,1});
		chromosome_length = individual.getChromosome().length;
		
		/*for there is only one cycle all genes should change*/
		number_of_genes_mutated = 9;
	}
	
	
	public void testSuccessfulMutation() {
		assertEquals(individual.getChromosome().length, chromosome_length);		
		
		mutation = new EvPermutationCycleRotationMutation(1.0);
		
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
}
