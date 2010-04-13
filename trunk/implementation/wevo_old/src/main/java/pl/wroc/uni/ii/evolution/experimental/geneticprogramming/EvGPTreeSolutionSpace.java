package pl.wroc.uni.ii.evolution.experimental.geneticprogramming;

// TODO this class needs comments and javadocs...

import java.util.GregorianCalendar;
import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals.EvGPTree;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals.EvGPType;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * @author Zbigniew Nazimek, Donata Malecka
 */
public class EvGPTreeSolutionSpace implements EvSolutionSpace<EvGPTree> {

  private static final long serialVersionUID = -3046898772712402723L;

  private EvObjectiveFunction<EvGPTree> objective_function;

  protected EvRandomizer rand =
      new EvRandomizer(new GregorianCalendar().getTimeInMillis());

  protected int max_depth = 5;


  public EvGPTreeSolutionSpace(
      EvObjectiveFunction<EvGPTree> objective_function, int max_depth) {
    this.max_depth = max_depth;
    setObjectiveFuntion(objective_function);
  }


  /**
   * @deprecated objective function must be first in the contructor, plz. use
   *             the contructor with correct order
   */
  public EvGPTreeSolutionSpace(int max_depth,
      EvObjectiveFunction<EvGPTree> objective_function) {
    this.max_depth = max_depth;
    setObjectiveFuntion(objective_function);
  }


  public EvGPTreeSolutionSpace(EvObjectiveFunction<EvGPTree> objective_function) {
    this(5, objective_function);
  }


  public boolean belongsTo(EvGPTree individual) {
    return false;
  }


  public Set<EvSolutionSpace<EvGPTree>> divide(int n) {
    return null;
  }


  public Set<EvSolutionSpace<EvGPTree>> divide(int n, Set<EvGPTree> p) {
    return null;
  }


  public EvGPTree generateIndividual() {
    int iter = 1;
    return generateIndividual(iter);

  }


  private EvGPTree generateIndividual(int i) {

    if (i < max_depth) {

      int r = rand.nextInt(EvGPType.values().length - 1);
      EvGPTree gpt =
          new EvGPTree(EvGPType.values()[r], rand.nextDouble(), rand.nextInt());

      if (EvGPType.values()[r] != EvGPType.CONSTANT
          || EvGPType.values()[r] != EvGPType.TABLE_ELEMENT) {
        if (EvGPType.values()[r] != EvGPType.COS
            || EvGPType.values()[r] != EvGPType.SIN
            || EvGPType.values()[r] != EvGPType.TAN) { // unar
          gpt.setLeftSubTree(generateIndividual(i + 1));
          gpt.setRightSubTree(generateIndividual(i + 1));
          gpt.setObjectiveFunction(objective_function);
          return gpt;
        } // unar
        gpt.setRightSubTree(generateIndividual(i + 1));
        gpt.setObjectiveFunction(objective_function);
        return gpt;
      }
      gpt.setObjectiveFunction(objective_function);
      return gpt;
    }

    return new EvGPTree(EvGPType.CONSTANT, rand.nextDouble(), rand.nextInt());

  }


  public EvGPTree takeBackTo(EvGPTree individual) {

    return null;
  }


  /**
   * {@inheritDoc}
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvGPTree> objective_function) {
    this.objective_function = objective_function;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvGPTree> getObjectiveFuntion() {
    return objective_function;
  }

}
