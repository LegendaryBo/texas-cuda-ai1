package pl.wroc.uni.ii.evolution.engine.individuals;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Class for natural number individual with fixed length.
 * 
 * @author Jarek Fuks, Kamil Dworakowski
 */
public class EvNaturalNumberVectorIndividual extends EvIndividual {

  private static final long serialVersionUID = -293840929375451L;

  private int[] vector;


  /**
   * @param d fixed length of individual
   */
  public EvNaturalNumberVectorIndividual(int d) {
    vector = new int[d];
  }


  /**
   * @param source table for vector
   */
  public EvNaturalNumberVectorIndividual(int... table) {
    vector = table.clone();
  }


  /**
   * Gets value of i-th bit.
   * 
   * @param i position in chromosome
   * @return value of i-th position
   */
  public int getNumberAtPosition(int i) {
    return vector[i];
  }


  /**
   * Sets value at a specific position in chromosme
   * 
   * @param i position in chromosome
   * @param b desired value of this position
   */
  public void setNumberAtPosition(int i, int b) {
    this.invalidate();
    vector[i] = b;
  }


  /**
   * Gets individual dimension.
   * 
   * @return length of chromosome
   */
  public int getDimension() {
    return vector.length;
  }


  public String toString() {
    StringBuffer output = new StringBuffer();
    for (int i = 0; i < vector.length; i++) {
      if (i < vector.length - 1) {
        output.append(vector[i] + ",");
      } else {
        output.append(vector[i]);
      }
    }
    return output.toString();
  }


  public EvNaturalNumberVectorIndividual clone() {
    EvNaturalNumberVectorIndividual n1 =
        new EvNaturalNumberVectorIndividual(vector);
    for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
      n1.addObjectiveFunction(this.getObjectiveFunction(i));
      if (this.isEvaluated(i)) {
        n1.assignObjectiveFunctionValue(getObjectiveFunctionValue(i), i);
      }
    }
    return n1;
  }


  /**
   * Check if individual contains given value in his natural number vector
   * 
   * @param value
   * @return true if individual contains given value, false if it doesn't
   */
  public boolean hasValue(int value) {
    for (int val : vector)
      if (val == value)
        return true;
    return false;
  }
}
