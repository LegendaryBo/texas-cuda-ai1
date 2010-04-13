package pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * This is a class which delay evaluation of given objective function (designed
 * for tests)
 * 
 * @author Kacper Gorski
 * @param <T>
 */
public class EvLongFunction<T extends EvIndividual> implements
    EvObjectiveFunction<T> {

  private static final long serialVersionUID = -8041645554510525440L;

  private EvObjectiveFunction<T> fun;

  private long delay;


  public EvLongFunction(EvObjectiveFunction<T> fun, long delay) {
    this.fun = fun;
    this.delay = delay;
  }


  public double evaluate(T individual) {

    try {
      Thread.sleep(delay);
    } catch (InterruptedException e) {
    }

    return fun.evaluate(individual);
  }

}
