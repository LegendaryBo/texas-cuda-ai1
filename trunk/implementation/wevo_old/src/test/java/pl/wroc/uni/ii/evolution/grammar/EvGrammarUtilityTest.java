package pl.wroc.uni.ii.evolution.grammar;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.grammar.EvGrammarEvolutionUtility;
import junit.framework.TestCase;

public class EvGrammarUtilityTest extends TestCase
{
	public void testKnaryToNaturalConversion() throws EvGrammarEvolutionException {		
		EvKnaryIndividual kn_ind = new EvKnaryIndividual(20, 2);		
		
		int[] genes = new int[] {1, 0, 1, 1,   // 11
								 1, 1, 0, 0,   // 12
								 1, 1, 1, 1,   // 15
								 0, 0, 1, 0,   // 2
								 0, 1, 1, 1};  // 7
		
		for (int i = 0; i < 20; i++)
			kn_ind.setGene(i, genes[i]);
		
		EvNaturalNumberVectorIndividual nn_ind 
			= EvGrammarEvolutionUtility.Convert(kn_ind, 4);
		
		assertEquals(nn_ind.getNumberAtPosition(0), 11);
		assertEquals(nn_ind.getNumberAtPosition(1), 12);
		assertEquals(nn_ind.getNumberAtPosition(2), 15);
		assertEquals(nn_ind.getNumberAtPosition(3), 2);
		assertEquals(nn_ind.getNumberAtPosition(4), 7);
	}
	
	public void testGrammaticGeneration() 
		throws EvBNFParsingException, EvGrammarEvolutionException {
		
		String str = "<S>" + '\n'
		+ "<S>::=<A>|1<S>0" + '\n'
		+ "<A>::=2" + '\n';

		EvBNFParser parser = new EvBNFParser(str);
		EvBNFGrammar grammar = parser.parse();
		
		int[] vals = new int[] {3, 7, 9, 2, 10}; // for word 1112000
		
		String text = EvGrammarEvolutionUtility.Generate(vals, grammar);
		assertEquals("1112000", text);
	}	
}