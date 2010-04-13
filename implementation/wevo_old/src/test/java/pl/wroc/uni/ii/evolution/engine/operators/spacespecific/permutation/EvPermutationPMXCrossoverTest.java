package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import junit.framework.TestCase;

/**
 * 
 * Class testing EvPermutationPMXCrossover
 * 
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 * @author Karol Asgaroth Stosiek (szymek.fogiel@gmail.com)
 *
 */
public class EvPermutationPMXCrossoverTest extends TestCase {	
	private int chromosome_length;
	private int segment_beginning;
	private int segment_length;
	private boolean exception_thrown;
	
	private EvPermutationIndividual parent1;
	private EvPermutationIndividual parent2;
	
	private List<EvPermutationIndividual> parents;
	
	private EvPermutationPMXCrossover crossover;
	
	
	public void setUp() {
		chromosome_length = 9;
		segment_beginning = 3;
		segment_length = 3;
		exception_thrown = false;
		
		parent1 = new EvPermutationIndividual(new int[] {1,2,3,4,5,6,7,8,9});
		parent2 = new EvPermutationIndividual(new int[] {4,1,2,8,7,6,9,3,5});
		
		parents = new ArrayList<EvPermutationIndividual>();
		parents.add(parent1);
		parents.add(parent2);
	}
	
	
	public void testSuccessfulCrossover() {
		assertEquals(parent1.getChromosome().length, chromosome_length);
		assertEquals(parent2.getChromosome().length, chromosome_length);
		
		crossover = new EvPermutationPMXCrossover
			(segment_beginning, segment_length);
		
		List<EvPermutationIndividual> children = crossover.combine(parents);
		
		EvPermutationIndividual child1 = children.get(0);
		EvPermutationIndividual child2 = children.get(1);
		
		int[] proper_child1_chromosome = new int[] {1,2,3,8,7,6,5,4,9};
		int[] proper_child2_chromosome = new int[] {8,1,2,4,5,6,9,3,7};
		
		for (int i = 0; i < chromosome_length; i++) {
			assertEquals(child1.getGeneValue(i), proper_child1_chromosome[i]);
			assertEquals(child2.getGeneValue(i), proper_child2_chromosome[i]);
		}
	}
	
	
	public void testTooSmallSegmentBeginningParameter() {						
		try {
			crossover = new EvPermutationPMXCrossover(-1, 1);
		} catch (IllegalArgumentException e) {
			exception_thrown = true;
		}
			
		assertEquals(exception_thrown,true);
		
	}
	
	
	public void testTooBigSegmentLengthParameter() {		
		exception_thrown = false;
		
		crossover = new EvPermutationPMXCrossover(segment_beginning, 10);
		
		try {
			crossover.combine(parents);
		} catch (IllegalArgumentException e) {
			exception_thrown = true;
		}
		
		assertEquals(exception_thrown,true);		
		
	}
		
	
	public void testTooBigSegmentBeginningParameter() {
		exception_thrown = false;
		
		crossover = new EvPermutationPMXCrossover(10, 1);
		
		try {
			crossover.combine(parents);
		} catch (IllegalArgumentException e) {
			exception_thrown = true;
		}
		
		assertEquals(exception_thrown,true);
	}
}
