package pl.wroc.uni.ii.evolution.utils;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import Jama.Matrix;
import junit.framework.TestCase;

/**
 * Tests for EvShrinkageCovarianceEstimator class.
 * 
 * @author Szymon Fogiel (szymek.fogiel@gmail.com)
 *
 */
public class EvShrinkageCovarianceEstimatorTest extends TestCase {
  
  /** Instance for tests. */
  private EvShrinkageCovarianceEstimator covariance_estimator;
  
  /** Population for tests. */
  private EvPopulation<EvRealVectorIndividual> population;
  
  /**
   * Sets up test.
   */
  public void setUp() {
    covariance_estimator = new EvShrinkageCovarianceEstimator();
    population = new EvPopulation<EvRealVectorIndividual>();
  }
      
  
  /**
   * Tests whether covariance matrix has positive determinants
   * for all it's square submatrixes.
   */
  public void testCountCovarianceMatrix() {
    int population_size = 50;
    int dimension = 20;  
    
    EvRealVectorIndividual individual; 
    
    for (int i = 0; i < population_size; i++) {
      individual = new EvRealVectorIndividual(dimension);
      for (int j = 0; j < dimension; j++) {
        individual.setValue(j, EvRandomizer.INSTANCE.nextDouble());
      }
      population.add(individual.clone());
    }    
    
    covariance_estimator.calculateCovarianceMatrix(population);
    
    Matrix matrix =
      new Matrix(covariance_estimator.getEstimatedCovarianceMatrix());
    
    for (int i = 1; i < dimension; i++) {
      assertTrue(matrix.getMatrix(0, i, 0, i).det() > 0);
    }
  }
}
