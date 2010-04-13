package pl.wroc.uni.ii.evolution.engine.prototype;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;

/**
 * A operator that makes a new population from those of parents and children.
 * Used, for example, in SGA.
 * 
 * @author Kamil Dworakowski (kamil.dworakowski@gmail.com)
 * @param <T> - type of individuals the operator works on
 */
public interface EvReplacement<T extends EvIndividual> {

  /**
   * Applies a replacement operator.
   * 
   * @param parents -- population of parents
   * @param children -- population of children
   * @return new population
   */
  EvPopulation<T> apply(EvPopulation<T> parents, EvPopulation<T> children);
}
