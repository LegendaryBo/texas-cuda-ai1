package pl.wroc.uni.ii.evolution.grammar;

import java.util.Hashtable;
import java.util.Vector;

/**
 * Class used to parse String to grammar.
 * 
 * @author Marta Stanska (martastanska@gmail.com)
 */
public class EvBNFParser {
  String str;


  /**
   * Creates instance of BNF parser
   * 
   * @param str_ String representation of grammar we want to extract
   */
  public EvBNFParser(String str_) {
    str = str_;
  }


  /**
   * Parses string given in grammar
   * 
   * @return new BNF grammar
   * @throws EvBNFParsingException in case of incorrect format
   */
  public EvBNFGrammar parse() throws EvBNFParsingException {
    String startSymbol = "";
    Hashtable<String, Vector<String>> productions =
        new Hashtable<String, Vector<String>>();

    int i = 0;
    char a = str.charAt(i);
    String nonterminalSymbol = "";
    Vector<String> choices = new Vector<String>();

    // reading the start symbol
    while (a != '\n') {
      Character c = new Character(a);
      startSymbol = startSymbol + c.toString();
      i++;
      a = str.charAt(i);
    }
    i++;
    a = str.charAt(i);

    // reading the productions
    while (i < str.length() - 2) {
      // reading the left side of production
      while (a != ':') {
        Character c = new Character(a);
        nonterminalSymbol = nonterminalSymbol + c.toString();
        i++;
        a = str.charAt(i);
      }

      i++;
      a = str.charAt(i);

      if (a != ':')
        throw new EvBNFParsingException();
      i++;
      a = str.charAt(i);

      if (a != '=')
        throw new EvBNFParsingException();
      i++;
      a = str.charAt(i);

      // reading the right side of production
      while (a != '\n') {
        String choice = "";
        // reading choice
        while (a != '|' && a != '\n') {
          Character c = new Character(a);
          choice = choice + c.toString();
          i++;
          a = str.charAt(i);
        }

        choices.add(choice);
        choice = "";
        if (a == '\n')
          break;
        i++;
        a = str.charAt(i);
      }

      if (i < str.length() - 2) {
        i++;
        a = str.charAt(i);
      }

      Vector<String> choices_ = new Vector<String>(choices);
      productions.put(nonterminalSymbol, choices_);
      nonterminalSymbol = "";
      choices.clear();
    }

    EvBNFGrammar grammar = new EvBNFGrammar(startSymbol, productions);
    return grammar;
  }
}