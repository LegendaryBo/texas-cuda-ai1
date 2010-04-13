package pl.wroc.uni.ii.evolution.objectivefunctions.grammar;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.grammar.EvBNFGrammar;
import pl.wroc.uni.ii.evolution.grammar.EvGrammarEvolutionException;
import pl.wroc.uni.ii.evolution.grammar.EvGrammarEvolutionUtility;

/**
 * @author Marta Stanska (martastanska@gmail.com)
 */
public class EvGrammarPattern implements EvObjectiveFunction<EvKnaryIndividual> {

  private static final long serialVersionUID = -4839873819742201982L;

  private EvBNFGrammar grammar;

  private String pattern;

  private int codonSize;


  /**
   * Constructs class EvGEHammingDistance
   * 
   * @param grammar_
   * @param pattern_
   * @param codonSize_
   */
  public EvGrammarPattern(EvBNFGrammar grammar_, String pattern_, int codonSize_) {
    super();
    grammar = grammar_;
    pattern = pattern_;
    codonSize = codonSize_;
  }


  /**
   * Evaluates individual
   * 
   * @param individual_
   * @result hamming distance between pattern and individual
   */
  public double evaluate(EvKnaryIndividual individual_) {
    int hammingDistance = 0;

    try {
      EvNaturalNumberVectorIndividual individual =
          EvGrammarEvolutionUtility.Convert(individual_, codonSize);
      String result = EvGrammarEvolutionUtility.Generate(individual, grammar);

      if (!EvGrammarEvolutionUtility.isTerminalWord(result))
        return -1;

      if (result.length() == pattern.length())
        for (int i = 0; i < pattern.length(); i++)
          if (result.charAt(i) == pattern.charAt(i))
            hammingDistance++;

      return hammingDistance;

    } catch (EvGrammarEvolutionException ex) {
      return -1;
    }
  }
}