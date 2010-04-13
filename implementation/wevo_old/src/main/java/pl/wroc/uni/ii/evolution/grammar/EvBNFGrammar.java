package pl.wroc.uni.ii.evolution.grammar;

import java.util.Hashtable;
import java.util.Vector;

/**
 * Class used for keeping the grammar.
 * 
 * @author Marta Stanska (martastanska@gmail.com)
 * @author Konrad Drukala (heglion@gmail.com)
 */
public class EvBNFGrammar {

  private String startSymbol;

  private Hashtable<String, Vector<String>> productions;


  /**
   * Creates instance of class representation for BNF grammar
   * 
   * @param startSymbol_ starting symbol of grammar
   * @param productions_ collection of productions
   */
  EvBNFGrammar(String startSymbol_,
      Hashtable<String, Vector<String>> productions_) {
    startSymbol = startSymbol_;
    productions = productions_;
  }


  /**
   * Getter for starting symbol
   * 
   * @return starting symbol
   */
  public String getStartSymbol() {
    return this.startSymbol;
  }


  /**
   * @param nonterminal Nonterminal for which we want productions
   * @return vector of possible choices
   * @throws EvBNFException in case of passing non-existant nonterminal
   */
  public Vector<String> getChoices(String nonterminal)
      throws EvGrammarEvolutionException {
    if (productions.containsKey(nonterminal))
      return productions.get(nonterminal);
    else
      throw new EvGrammarEvolutionException(
          "Production for nonexistetnt nonterminal requested");
  }
}
