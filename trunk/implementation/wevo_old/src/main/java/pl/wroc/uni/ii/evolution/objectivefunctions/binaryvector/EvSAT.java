package pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A SAT formula objective function class. This objective function calculates
 * number of satisfied clausules in SAT formula. An BinaryVectorIndividual codes
 * the variable assignment - i-th bit set to 1 means that i-th variable is true.
 * The class can generate random SAT formulas with the given parameters, from
 * the given seed. This means the same seed value will provide the same formula
 * if the given parameters are also the same.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */
public class EvSAT implements EvObjectiveFunction<EvBinaryVectorIndividual> {

  /**
   * 
   */
  private static final long serialVersionUID = 129800845815902201L;

  /**
   * Stores the formula.
   */
  private int[][] formula;

  /**
   * Number of variables in the formula.
   */
  private int variables_number;


  /**
   * Constructor, creates a random SAT formula. <br>
   * For example to generate 3SAT random formula use minvar = maxvar = 3.
   * 
   * @param clausules - number of clausules in the formula
   * @param variables - number of variables in the formula
   * @param minvar - minimal number of literals in each clausule
   * @param maxvar - maximal number of literals in each clausule
   * @param seed - seed for the random number generator
   */
  public EvSAT(final int clausules, final int variables, 
      final int minvar, final int maxvar, final long seed) {
    
    final double initProbability = 0.5;

    EvRandomizer randomizer = new EvRandomizer(seed);
    formula = new int[clausules][];

    for (int i = 0; i < formula.length; i++) {
      formula[i] = new int[randomizer.nextInt(maxvar - minvar + 1) + minvar];
      for (int j = 0; j < formula[i].length; j++) {
        if (randomizer.nextDouble() < initProbability) {
          formula[i][j] = randomizer.nextInt(variables) + 1;
        } else {
          formula[i][j] = -randomizer.nextInt(variables) - 1;
        }
      }
    }

    countVariablesNumber();
  }


  /**
   * Constructor, creates a given formula. <br>
   * The given formula is an array of clausules, which are arrays of literals.
   * formula[m][var] stores a SAT formula. formula[i] is an array of i-th
   * clausule literal. formula[i][j] is an j-th literal of i-th clausule. If
   * literal is not a negated variable then there is positive integer index of
   * it. If literal is a negated variable then there is negated integer index of
   * it. <br>
   * For example: {{1, 2}, {1, -2, -3}} corresponds to SAT formula: (a || b) &&
   * (a || !b || !c)
   * 
   * @param formula_ - the given formula stored as an array of clausules;
   */
  public EvSAT(final int[][] formula_) {
    this.formula = formula_;
    countVariablesNumber();
  }


  /**
   * Constructor, parses a string with formula in popular cnf format. This
   * format is used for example in (http://www.satcompetition.org). <br>
   * The format defines one clausule per one line ended with 0, there can be
   * also comments - lines with letter 'c' as first character. This constructor
   * can also parse simplified cnf format which does not require p defining line
   * and zero 0, also there can be extra spaces and extra lines (but with a
   * letter as first character to make the difference from clausule definition).
   * For example: <br>
   * c it is a comment here<br>
   * p cnf 3 2<br>
   * 1 2 0<br>
   * 1 -2 -3 0<br>
   * <br>
   * can be also parsed in simplified version:<br>
   * 1 2<br>
   * 1 -2 -3<br>
   * <br>
   * both of them is equivalent and defines the SAT formula:<br>
   * (a | b) & (a | !b | !c)<br>
   * 
   * @param cnfformula - the given formula as string in cnf format
   */
  public EvSAT(final String cnfformula) {

    String[] lines = cnfformula.split("\n");

    // Count the clausules
    int m = 0;
    for (int i = 0; i < lines.length; i++) {
      if (!(lines[i].length() == 0) 
          && !Character.isLetter(lines[i].charAt(0))) {
        m++;
      }
    }

    // Create the formula
    formula = new int[m][];

    int clausule = 0;
    for (int i = 0; i < lines.length; i++) {
      if (!(lines[i].length() == 0) 
          && !Character.isLetter(lines[i].charAt(0))) {

        String[] columns = lines[i].split(" ");

        // Count and parse the variables
        int n = 0;
        int[] variables = new int[columns.length];
        for (int j = 0; j < columns.length; j++) {
          try {
            variables[j] = Integer.parseInt(columns[j]);
            if (variables[j] != 0) {
              n++;
            }
          } catch (NumberFormatException e) {
            variables[j] = 0;
          }
        }

        // Create the clausule
        formula[clausule] = new int[n];

        int variable = 0;
        for (int j = 0; j < columns.length; j++) {
          if (variables[j] != 0) {
            formula[clausule][variable] = variables[j];
            variable++;
          }
        }

        clausule++;
      }
    }

    countVariablesNumber();
  }


  /**
   * Counts number of variables in the formula.
   */
  protected void countVariablesNumber() {

    variables_number = 0;

    for (int i = 0; i < formula.length; i++) {
      for (int j = 0; j < formula[i].length; j++) {
        if (formula[i][j] > 0) {
          if (formula[i][j] > variables_number) {
            variables_number = formula[i][j];
          }
        } else {
          if (-formula[i][j] > variables_number) {
            variables_number = -formula[i][j];
          }
        }
      }
    }
  }


  /**
   * Gets the variables number in the formula.
   * 
   * @return variables number
   */
  public int getVariablesNumber() {
    return variables_number;
  }


  /**
   * Gets the formula.
   * 
   * @return formula stored in array of clausules
   */
  public int[][] getFormula() {
    return formula;
  }


  /**
   * Sets the formula.
   * 
   * @param formula_ - formula stored as an array of clausules
   */
  public void setFormula(final int[][] formula_) {
    this.formula = formula_;
    countVariablesNumber();
  }


  /**
   * {@inheritDoc}
   */
  public double evaluate(final EvBinaryVectorIndividual individual) {

    // Check if individuals specifies required number of variables
    if (individual.getDimension() < variables_number) {
      throw new IllegalArgumentException(
          "The given individual does not specify all required variables");
    }

    // Count the satified clausules
    int satisfied_number = 0;

    for (int i = 0; i < formula.length; i++) {
      // Check if i-th clausule is satisfied
      for (int j = 0; j < formula[i].length; j++) {
        if (formula[i][j] > 0) {
          if (individual.getGene(formula[i][j] - 1) != 0) {
            satisfied_number++;
            break;
          }
        } else {
          if (individual.getGene(-formula[i][j] - 1) == 0) {
            satisfied_number++;
            break;
          }
        }
      }
    }

    // The objective function value is number of satisfied formulas
    return satisfied_number;
  }

}