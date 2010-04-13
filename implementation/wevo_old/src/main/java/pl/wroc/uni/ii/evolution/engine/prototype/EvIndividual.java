/**
 * 
 */
package pl.wroc.uni.ii.evolution.engine.prototype;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Abstract class for individuals with more than one objective function.
 * NOTE! If you want to implement your own individuals, you must also implement
 * a class extending object EvSolutionSpace, otherwise you won't be able to use
 * wEvo library. And you can use algorithms, operators, etc that are able to 
 * use more than one objective functions. 
 * 
 * @author Marcin Brodziak (marcin@nierobcietegowdomu.pl)
 * @author Tomasz Kozakiewicz (quzzaq@gmail.com)
 * @author Kacper Gorski (admin@34all.org)
 * Multiobjective functionality:
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 */
public abstract class EvIndividual  
    implements Serializable, Comparable, Cloneable {
  //off VisibilityModifier
  /** If individual to be compared has different length of obj. fun. */
  private static final int LENGHT_OBJECTIVE_FUNCTION_ERROR = 100;
  
  /** Values of objective functions, used for memorization. */
  private double[] objective_function_value = new double[0];
  
  /** Status if objective function is calculated or not. */
  private boolean[] objective_calculated = new boolean[0];
  
  /** Status if all objective functions are calculated or not. */
  private boolean objectives_calculated = false;
  
  /** Objective function associated with this individual. */
  private ArrayList<EvObjectiveFunction> objective_function 
      = new ArrayList<EvObjectiveFunction>();
  //on VisibilityModifier
  
  /**
   * Return all objective functions added to this individual.
   * @return All objective functions. 
   */
  public ArrayList<EvObjectiveFunction> getObjectiveFunctions() {
    return this.objective_function;
  }
  
  /**
   * Get first objective function.
   * @return the objective_function
   * @deprecated Recommend to use getObjectiveFunction(int)
   */
  @Deprecated
  public EvObjectiveFunction getObjectiveFunction() {
    return getObjectiveFunction(0);
  }
  
  /**
   * getter.
   * @param i objective_function index
   * @return the objective_function
   */
  public EvObjectiveFunction getObjectiveFunction(final int i) {
    try {
      checkIndex(i);
    } catch (Exception e) {
      //e.printStackTrace();
      return null;
    }
    return objective_function.get(i);
  }

  /**
   * setter.
   * @param objective_function_ the objective_function to set
   */
  public void setObjectiveFunctions(final ArrayList<EvObjectiveFunction> 
      objective_function_) {
    this.objective_function = objective_function_;
    this.objective_function_value = new double[objective_function.size()];
    this.objective_calculated = new boolean[objective_function.size()];
    invalidate();
  }
  
  /**
   * Remove all objective function and set new one on the first positon.
   * @param objective_function_ new objective function.
   * @deprecated Recommend to use setObjectiveFunction(EvObjectiveFunction,int)
   */
  public void setObjectiveFunction(final EvObjectiveFunction 
     objective_function_) {
    this.objective_function = new ArrayList<EvObjectiveFunction>();
    this.objective_function.add(objective_function_);
    this.objective_function_value = new double[objective_function.size()];
    this.objective_calculated = new boolean[objective_function.size()];
    invalidate();
  }
  
  /**
   * Set new objective function on position i.
   * @param objective_function_ new Objective function
   * @param i Index of old objective function to be changed
   */
  public void setObjectiveFunction(final EvObjectiveFunction 
      objective_function_, final int i) {
    try {
      checkIndex(i);
    } catch (Exception e) {
      e.printStackTrace();
    }
    this.objective_function.set(i, objective_function_);
    this.objective_calculated[i] = false;
  }

  /**
   * Add new objective function to the end of list's function.
   * @param objective_function_ new objective function
   */
  public void addObjectiveFunction(final EvObjectiveFunction
      objective_function_) {
    this.objective_function.add(objective_function_);
    double[] new_obj_values = new double[objective_function.size()];
    boolean[] new_obj_calcuated = new boolean[objective_function.size()];
    for (int i = 0; i < objective_calculated.length; i++) {
      new_obj_values[i] = objective_function_value[i];
      new_obj_calcuated[i] = objective_calculated[i];
    }
    new_obj_calcuated[new_obj_calcuated.length - 1] = false;
    objective_calculated = new_obj_calcuated;
    objective_function_value = new_obj_values;
    objectives_calculated = false;
  }
  
  /**
   * getter.
   * @return the objective_function_value
   * @deprecated Recommend to use getObjectiveFunctionValue(int)
   */
  public double getObjectiveFunctionValue() {
    return getObjectiveFunctionValue(0);
  }
  
  /**
   * Get objective function value at position i.
   * @param i Objective function position.
   * @return Value of objective function at position i.
   */
  public double getObjectiveFunctionValue(final int i) {
    try {
      checkIndex(i);
    } catch (Exception e) {
      e.printStackTrace();
    }
    if (!objective_calculated[i]) {
      objective_function_value[i] = objective_function.get(i).evaluate(this);
      objective_calculated[i] = true;
      if (!this.objectives_calculated) {
        checkEvaluation();
      }
    }
    return objective_function_value[i];
  }
  
  /**
   * Should be called after evaluating objective function to this individual. It
   * stores it's value and returns them in further calls of
   * getObjectiveFunctionValue() (unless the state of the individual hasen't
   * changed).
   *
   * Note: This method is used only when retrieving values of objective function
   * from external sources (i.e. computed on distributed nodes). Should never
   * be called on its own under normal circumstances.
   *
   * @param value value of objective function
   * @deprecated Recommend to use assignObjectiveFunctionValue(double,int).
   */
  @Deprecated
  public final void assignObjectiveFunctionValue(final double value) {
    assignObjectiveFunctionValue(value, 0);
  }
  
  /**
   * Should be called after evaluating objective function to this individual. It
   * stores it's value and returns them in further calls of
   * getObjectiveFunctionValue() (unless the state of the individual hasen't
   * changed).
   *
   * Note: This method is used only when retrieving values of objective function
   * from external sources (i.e. computed on distributed nodes). Should never
   * be called on its own under normal circumstances.
   *
   * @param i index of objective function
   * @param value value of objective function
   */
  public final void assignObjectiveFunctionValue(final double value,
      final int i) {
    try {
      checkIndex(i);
    } catch (Exception e) {
      e.printStackTrace();
    }
    objective_function_value[i] = value;
    objective_calculated[i] = true;
    if (!this.objectives_calculated) {
      checkEvaluation();
    }
  }
  
  /**
   * Must be marked when any kind of modification of internal state that changes
   * value of objective function is calculated. (for example, changing bits
   * value in binary vector individuals)
   */
  protected void invalidate() {
    objectives_calculated = false;
    for (int i = 0; i < this.objective_calculated.length; i++) {
      this.objective_calculated[i] = false;
    }
  }
  
  /**
   * Check if every objective function is calculated.
   * If it's true then set isValidated function on true. 
   */
  protected void checkEvaluation() {
    boolean allTrue = true;
    for (int i = 0; i < this.objective_calculated.length; i++) {
      if (!this.objective_calculated[i]) {
        allTrue = false;
      }
    }
    if (allTrue) {
      this.objectives_calculated = true;
    }
  }  
  
  /**
   * Tells whether the individual knows it's objective function value.
   *
   * @return true if objective function value is known, false if it's not
   * @deprecated Recommend to use isEvaluated()
   */
  @Deprecated 
  public final boolean isObjectiveFunctionValueCalculated() {
    return isEvaluated();
  }
  
  /**
   * Tells whether the individual knows it's objective functions value.
   * @return true if objective function value is known, false if it's not
   */
  public boolean isEvaluated() {
    return objectives_calculated;
  }
  
  /**
   * Tells whether the individual knows it's objective function value.
   * @param i Index of objective function.
   * @return true if objective function value is known, false if it's not 
   */
  protected boolean isEvaluated(final int i) {
    try {
      checkIndex(i);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return objective_calculated[i];
  }
  
  /**
   * Compares two individuals and decides which of them has highest objective
   * functions value using Pareto criterion.<br>
   * Pareto criterion:<br>
   * If every objective functions values are higher than other individual then
   * this object is higher (P-optimum).
   * If every objective functions values are smaller than other individual then
   * the individual is higher.
   * <br>
   * 
   * @param o Object that we will be comparing to
   * @return <BR>
   *         -1 if this individual has smaller every obj. fun value<br>
   *         0 if their obj. values are equal<br>
   *         1 if this individual has every higher value<br>
   *         100 if amount of objective functions is different
   */
  public int compareTo(final Object o) {
    EvIndividual individual = (EvIndividual) o;
    if (objective_function.size() 
        != individual.getObjectiveFunctions().size()) {
      //throw new Exception("Length of objective functions isn't equal!");
      return LENGHT_OBJECTIVE_FUNCTION_ERROR;
    }
    //this individual has higher values of objective functions
    boolean thisHigher = true;  
    //this individual has smaller values of objective functions
    boolean thisSmaller = true;
    for (int i = 0; i < this.objective_function.size(); i++) {
      try {
        if (individual.getObjectiveFunctionValue(i) 
            > getObjectiveFunctionValue(i)) {
          thisHigher = false;
        }
      } catch (Exception e) {
        e.printStackTrace();
      }
      try {
        if (individual.getObjectiveFunctionValue(i) 
            < getObjectiveFunctionValue(i)) {
          thisSmaller = false;
        }
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
    
    if (thisHigher && !thisSmaller) {
      return 1;
    }
    if (thisSmaller && !thisHigher) {
      return -1;
    }
    return 0;
    
  }
  
  /**
   * Check if index of objective function isn't out of bond.
   * @param i Number of objective function which we want to get access.
   * @throws Exception Exception if index is wrong.
   */
  private void checkIndex(final int i) throws Exception {
    if (i >= objective_function.size() || i < 0) {
      throw new Exception("Index " + i
          + " is out of bond. Length of objective functions is"
          + objective_function.size());
    }
  }
  
  /**
   * Cloning of the object.
   * 
   * @return clone of this object
   */
  public abstract Object clone();
}
