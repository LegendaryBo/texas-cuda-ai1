package pl.wroc.uni.ii.evolution.objectivefunctions.grammar;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberIndividualTest;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.grammar.EvBNFGrammar;
import pl.wroc.uni.ii.evolution.grammar.EvBNFParser;
import pl.wroc.uni.ii.evolution.grammar.EvBNFParsingException;
import pl.wroc.uni.ii.evolution.grammar.EvGrammarEvolutionException;
import pl.wroc.uni.ii.evolution.grammar.EvGrammarEvolutionUtility;
import junit.framework.TestCase;


public class EvGrammarPatternTest extends TestCase {
	
	public void testGrammarPatternEvaluation() 
		throws EvBNFParsingException, EvGrammarEvolutionException {
		String str = "<S>" + '\n' + 
					 "<S>::=<A><A><A><A><A><A><A><A>" + '\n' + 
					 "<A>::=0|1" + '\n';
		String pattern = "10100101";
		
		int[] bits = {1, 1, 1,
				      0, 1, 1,
				      1, 1, 0,
				      1, 0, 1,
				      0, 0, 0,
				      1, 1, 0,
				      1, 1, 1,
				      0, 1, 0,
				      0, 0, 1
		};
		
		EvKnaryIndividual ekv = new EvKnaryIndividual(bits, 2);		
		
		EvBNFParser parser = new EvBNFParser(str);
		EvBNFGrammar grammar = parser.parse();
		
		EvGrammarPattern patternEvaluator 
			= new EvGrammarPattern(grammar, pattern, 3);	
		
		assertTrue(patternEvaluator.evaluate(ekv) == 8.0);		
	}
}
