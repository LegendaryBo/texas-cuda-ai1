package pl.wroc.uni.ii.evolution.distribution.clustering;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;

/**
 * Interface to all classes that are used for loading solution space from some
 * source
 * 
 * @author Marcin Golebiowski
 */
public interface EvSolutionSpaceLoader<T extends EvIndividual> {

  EvSolutionSpace<T> takeSubspace();


  boolean newSubspaceAvailable();


  Thread start();
}
