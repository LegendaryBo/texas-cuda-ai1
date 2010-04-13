package pl.wroc.uni.ii.evolution.engine.individuals;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Class for real vector individual with fixed length.
 * 
 * @author Kamil Dworakowski, Jarek Fuks
 */
public class EvRealVectorIndividual extends EvIndividual {

  private static final long serialVersionUID = 1L;

  protected double[] vector;


  /**
   * Constructs individual with specified number of values set to 0.0.<BR>
   * 
   * @param d fixed length of individual
   */
  public EvRealVectorIndividual(int d) {
    vector = new double[d];
  }


  /**
   * Facility constructor.
   * 
   * @param vector
   */
  public EvRealVectorIndividual(double[] vector) {
    this.vector = vector;
  }


  /**
   * Gets real value of i-th value.
   * 
   * @param i value position in chromosome
   * @return value of i-th value
   */
  public double getValue(int i) {
    return vector[i];
  }


  /**
   * Sets ith element to value.
   * 
   * @param ith position in chromosome
   * @param value real value to set
   */
  public void setValue(int i, double value) {
    this.invalidate();
    vector[i] = value;
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
      if (i != 0)
        output.append(" ");
      output.append(vector[i]);
    }
    return output.toString();
  }


  @Override
  public EvRealVectorIndividual clone() {
    double[] table = vector.clone();
    EvRealVectorIndividual v1 = new EvRealVectorIndividual(table);
    for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
      v1.addObjectiveFunction(this.getObjectiveFunction(i));
      if (this.isEvaluated(i)) {
        v1.assignObjectiveFunctionValue(getObjectiveFunctionValue(i), i);
      }
    }
    return v1;
  }

}
