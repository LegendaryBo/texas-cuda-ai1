package pl.wroc.uni.ii.evolution.sampleimplementation.students.sabinafabiszewska;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * @author Sabina Fabiszewska
 */
public class EvMyIndividual extends EvIndividual {

  /**
   * 
   */
  private static final long serialVersionUID = 6063465761483322347L;

  /**
   * 
   */
  private final boolean[] vector;


  /**
   * @param l length of vector
   */
  public EvMyIndividual(final int l) {
    vector = new boolean[l];
    for (int i = 0; i < vector.length; i++) {
      vector[i] = EvRandomizer.INSTANCE.nextBoolean();
    }
    setObjectiveFunction(new EvMyObjectiveFunction());
  }


  /**
   * @param i index of the bit (in vector)
   * @return value of the i-th bit in the vector
   */
  public boolean getBit(final int i) {
    return vector[i];
  }


  /**
   * @param i index of the bit (in vector)
   * @param value value to which the i-th bit will be changed
   */
  public void setBit(final int i, final boolean value) {
    vector[i] = value;
  }


  /**
   * @return dimension of vector
   */
  public int getDimension() {
    return vector.length;
  }


  /**
   * @return individual converted to string
   */
  @Override
  public String toString() {
    String str = "[ ";
    for (int i = 0; i < vector.length; i++) {
      if (i != 0) {
        str += ", ";
      }
      if (vector[i]) {
        str += "1";
      } else {
        str += "0";
      }
    }
    str += " ]";
    return str;
  }


  /**
   * @return cloned object
   */
  @Override
  public Object clone() {
    EvMyIndividual individual = new EvMyIndividual(vector.length);
    for (int i = 0; i < vector.length; i++) {
      individual.vector[i] = this.vector[i];
    }
    return individual;
  }

}