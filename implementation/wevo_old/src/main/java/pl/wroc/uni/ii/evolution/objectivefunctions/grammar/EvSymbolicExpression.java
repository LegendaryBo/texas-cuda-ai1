package pl.wroc.uni.ii.evolution.objectivefunctions.grammar;

import java.util.Random;

import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.grammar.EvGrammarEvolutionException;
import pl.wroc.uni.ii.evolution.grammar.EvBNFGrammar;
import pl.wroc.uni.ii.evolution.grammar.EvGrammarEvolutionUtility;

public class EvSymbolicExpression implements
    EvObjectiveFunction<EvNaturalNumberVectorIndividual> {

  private static final long serialVersionUID = 1303564647340422918L;

  private EvBNFGrammar grammar;

  private String expression;

  private String[] varNames;

  private int numberOfTests;

  private int min, max;

  private static Random randomizer = new Random();


  /**
   * Creates instance of evaluator designed for evaluation of symbolic
   * expressions
   * 
   * @param grammar BNF grammar for generating symbolic expression
   * @param expression expression to compare
   * @param varNames names of variables occurring in expression
   * @param numberOfTests number of comparisons to do
   */
  public EvSymbolicExpression(EvBNFGrammar grammar, String expression,
      String[] varNames, int numberOfTests) {
    this.grammar = grammar;
    this.expression = expression;
    this.varNames = varNames;
    this.numberOfTests = numberOfTests;
    this.min = -10;
    this.max = 10;
  }


  /**
   * Generates Java code responsible for evaluating given expression
   * 
   * @param my_expression expression to compare
   * @return java code (in string)
   */
  private String generateClassCode(String my_expression) {
    double[][] var_vals = new double[numberOfTests][varNames.length];

    for (int i = 0; i < numberOfTests; i++)
      for (int j = 0; j < varNames.length; j++)
        var_vals[i][j] =
            randomizer.nextDouble()
                * (randomizer.nextInt(this.max - this.min) + this.min);

    StringBuilder sb = new StringBuilder();
    sb.append("public class SEClass {\n");

    for (int i = 0; i < varNames.length; i++) {
      sb.append("double[] var_" + varNames[i] + ";\n" + "{");
      for (int j = 0; j < numberOfTests; j++) {
        sb.append(var_vals[i][j]);
        if (j != numberOfTests - 1)
          sb.append(",\n");
      }
      sb.append("};\n");
    }
    sb.append("\n\n\t");
    sb.append("public static void main(String[] args) {\n");
    for (int i = 0; i < varNames.length; i++)
      sb.append("\t\tdouble " + varNames[i] + ";\n");
    sb.append("\t\tint output = 0;\n\n");
    sb.append("\t\tfor (int i = 0; i < " + numberOfTests + "; i++) {\n");
    for (int i = 0; i < varNames.length; i++)
      sb.append("\t\t\t" + varNames[i] + " = var_" + varNames[i] + "[i];\n");

    sb.append("\n");
    sb.append("\t\t\tif ( (" + expression + ") == (" + my_expression
        + ") ) output++;\n");
    sb.append("\t\t}\n");
    sb.append("\t\treturn output;\n");
    sb.append("\t}\n");
    sb.append("}");

    return sb.toString();
  }


  public double evaluate(EvNaturalNumberVectorIndividual individual) {
    try {
      String my_expression =
          EvGrammarEvolutionUtility.Generate(individual, grammar);
      String classText = this.generateClassCode(my_expression);

      return compileAndRun(classText);
    } catch (EvGrammarEvolutionException egee) {
      return 0;
    }
  }


  /**
   * TODO (Konrad Drukala): Compiles and runs code passed in argument
   * 
   * @param code code to compile and execute
   * @return value returned by executed code
   */
  private int compileAndRun(String code) {
    return 0;
  }
}