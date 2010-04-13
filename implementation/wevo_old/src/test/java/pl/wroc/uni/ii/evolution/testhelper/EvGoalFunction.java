package pl.wroc.uni.ii.evolution.testhelper;

import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

public class EvGoalFunction implements EvObjectiveFunction<EvStunt> {

  private static final long serialVersionUID = 5160229699054820092L;

  public double evaluate(EvStunt individual) {
    return individual.getValue();
  }

}
