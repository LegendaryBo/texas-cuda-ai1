package pl.wroc.uni.ii.evolution.grammar;

import java.util.Vector;

/**
 * Class used for keeping production from a grammar.
 * 
 * @author Marta Stanska (martastanska@gmail.com)
 */
public class EvBNFProduction {

  private String leftSide;

  private Vector<String> rightSide;


  /**
   * Creates new instance of BNF production
   * 
   * @param leftSide_ left side of production
   * @param rightSide_ collection of possible choices
   */
  EvBNFProduction(String leftSide_, Vector<String> rightSide_) {
    leftSide = leftSide_;
    rightSide = rightSide_;
  }


  /**
   * @return number of possible choices
   */
  int numberOfChoices() {
    return rightSide.size();
  }


  /**
   * Encapsulated indexer for vector class
   * 
   * @param index index of requested element
   * @return requested choice
   */
  String getChoice(int index) {
    if (index < rightSide.size())
      return rightSide.elementAt(index);
    else
      return "";
  }


  /**
   * Checks whether given nonterminal symbol is same as in this instance of
   * production
   * 
   * @param nonterminalSymbol_ nonterminal to check
   * @return true if matches, false otherwise
   */
  boolean matchNonterminalSymbol(String nonterminalSymbol_) {
    return leftSide.equals(nonterminalSymbol_);
  }

}
