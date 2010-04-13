package pl.wroc.uni.ii.evolution.engine.operators.general.misc;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;

/**
 * This operator changes individuals in population so that all of them are in
 * given solution space
 * 
 * @author Marek Chrusciel, Michal Humenczuk
 */
public class EvTakeBackToSolutionSpace<T extends EvIndividual> implements
    EvOperator<T> {

  private EvSolutionSpace<T> space;


  /**
   * Basic constructor
   * 
   * @param space solution to which individuals are taken back
   */
  public EvTakeBackToSolutionSpace(EvSolutionSpace<T> space) {
    this.space = space;
  }


  /**
   * {@inheritDoc}}
   */
  public EvPopulation<T> apply(EvPopulation<T> population) {
    EvPopulation<T> new_population = new EvPopulation<T>();

    for (T ind : population) {
      if (!space.belongsTo(ind)) {
        T back = space.takeBackTo(ind);
        back.setObjectiveFunction(ind.getObjectiveFunction());
        new_population.add(back);
      } else {
        new_population.add(ind);
      }
    }
    return new_population;

  }

}
