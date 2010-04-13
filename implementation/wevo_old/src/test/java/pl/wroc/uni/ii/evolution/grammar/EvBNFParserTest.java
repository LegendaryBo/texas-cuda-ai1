package pl.wroc.uni.ii.evolution.grammar;

import junit.framework.TestCase;

/**
 * Class used to test parsing the grammar by EvBNFParserTest
 * 
 * @author Marta Stanska (martastanska@gmail.com)
 */

public class EvBNFParserTest extends TestCase {
	
	public void testParse() throws EvBNFParsingException, EvGrammarEvolutionException {
		String str = "<S>" + '\n'
					+ "<S>::=<A>|1<S>0" + '\n'
					+ "<A>::=2" + '\n';

		EvBNFParser parser = new EvBNFParser(str);
		EvBNFGrammar grammar = parser.parse();
		
		assertEquals(grammar.getStartSymbol(), "<S>");
		assertTrue(grammar.getChoices("<A>").contains("2"));
		assertTrue(grammar.getChoices("<S>").contains("<A>"));
		assertTrue(grammar.getChoices("<S>").contains("1<S>0"));		
	}
	
	public void testParseException() throws EvGrammarEvolutionException {
		String str = "<S>" + '\n' + "<S>:=1<S>|0";
		
		EvBNFParser parser = new EvBNFParser(str);
		try {
		EvBNFGrammar grammar = parser.parse();
		} catch(EvBNFParsingException ex) {
			assertTrue(true);
		}		
	}
}