/**
 * 
 */
package pl.wroc.uni.ii.evolution.grammar;

import java.util.Vector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;

/**
 * Contains converters designed for Grammar. Evolution
 * 
 * @author Konrad Drukala (heglion@gmail.com)
 */
public final class EvGrammarEvolutionUtility {
  /**
   * Converts EvKnaryIndividual to EvNaturalNumberIndividual using codonSize
   * genes for generating new gene
   * 
   * @param individual Individual for conversion
   * @param codonSize Number of genes used for generating one new gene
   * @return New individual based on input individual
   * @throws EvGrammarEvolutionException in case of not matching lengths of
   *         individual and codonSize
   */
  public static EvNaturalNumberVectorIndividual Convert(
      EvKnaryIndividual individual, int codonSize)
      throws EvGrammarEvolutionException {
    if (individual.getDimension() % codonSize != 0)
      throw new EvGrammarEvolutionException(
          "Individual dimension must be multiple of codon size, dimension = "
              + individual.getDimension() + ", codonSize = " + codonSize);

    int conv_base = individual.getMaxGeneValue();
    int[] output = new int[individual.getDimension() / codonSize];

    for (int i = 0; i < individual.getDimension() / codonSize; i++) {
      output[i] = 0;
      for (int j = 0; j < codonSize; j++)
        output[i] +=
            individual.getGene(codonSize * i + j)
                * ((int) Math.pow(conv_base, codonSize - 1 - j));

    }
    return new EvNaturalNumberVectorIndividual(output);
  }


  /**
   * Generates words using given values and given grammar
   * 
   * @param values values for word generation
   * @param grammar grammar on which we base generation process
   * @return word belonging to grammar
   * @throws EvBNFException in case of incorrectness or incompletness of grammar
   */
  public static String Generate(int[] values, EvBNFGrammar grammar)
      throws EvGrammarEvolutionException {
    String out = grammar.getStartSymbol();
    String target, replacment;
    Vector<String> choices;

    for (int i = 0; i < values.length; i++) {
      target = EvGrammarEvolutionUtility.FindLeftmostNonterminal(out);
      if (target == null)
        return out;
      choices = grammar.getChoices(target);
      replacment = choices.get(values[i] % choices.size());
      out = out.replaceFirst(target, replacment);
    }
    return out;
  }


  /**
   * Generates words using given individual and given grammar
   * 
   * @param values individual for word generation
   * @param grammar grammar on which we base generation process
   * @return word belonging to grammar
   * @throws EvBNFException in case of incorrectness or incompletness of grammar
   */
  public static String Generate(EvNaturalNumberVectorIndividual individual,
      EvBNFGrammar grammar) throws EvGrammarEvolutionException {
    int[] values = new int[individual.getDimension()];
    for (int i = 0; i < individual.getDimension(); i++)
      values[i] = individual.getNumberAtPosition(i);
    return Generate(values, grammar);
  }


  /**
   * Checks whether word is terminal
   * 
   * @param word to check
   * @return returns true if word is terminal, false in other case
   */
  public static boolean isTerminalWord(String word) {
    char c;
    for (int i = 0; i < word.length(); i++) {
      c = word.charAt(i);
      if (c == '>' || c == '<')
        return false;
    }
    return true;
  }


  /**
   * Finds first occurrence of nonterminal symbol (in brackets)
   * 
   * @param input string in which we want to find nonterminal
   * @return nonterminal if present, null otherwise
   * @throws EvGrammarEvolutionException in case of incorrect format
   */
  private static String FindLeftmostNonterminal(String input)
      throws EvGrammarEvolutionException {
    if (isTerminalWord(input))
      return null;
    int i = 0;
    int j = 0;

    for (; i < input.length(); i++)
      if (input.charAt(i) == '<')
        break;

    for (j = i; j < input.length(); j++)
      if (input.charAt(j) == '>')
        break;

    String out = input.substring(i, j + 1);
    if (out.charAt(0) == '<' && out.charAt(out.length() - 1) == '>')
      return out;
    else
      throw new EvGrammarEvolutionException(
          "Input string has incorrect format: input = " + input + " output = "
              + out);
  }
}