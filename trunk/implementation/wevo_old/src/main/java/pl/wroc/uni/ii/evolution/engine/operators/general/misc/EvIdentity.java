package pl.wroc.uni.ii.evolution.engine.operators.general.misc;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Returns a copy of population
 * 
 * @author Zbigniew Nazimek, Jarek Fuks
 */
public class EvIdentity<T extends EvIndividual> implements EvOperator<T> {

  public EvPopulation<T> apply(EvPopulation<T> population) {
    return new EvPopulation<T>(population);
  }
}
