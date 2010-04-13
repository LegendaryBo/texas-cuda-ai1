package pl.wroc.uni.ii.evolution.utils;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;


/**
 * Class for estimating covariance matrix (and mean vector)
 * for a population of EvRealVector individuals.
 * 
 * @author Szymon Fogiel (szymek.fogiel@gmail.com)
 *
 */
public class EvShrinkageCovarianceEstimator {    
  
  /** Estimated covariance matrix. */
  private double[][] covariance_matrix; 
  
  /** Estimated mean vector. */
  private double[] mean_vector; 
  
  /** Sample population. */
  private EvPopulation<EvRealVectorIndividual> population;
  
  /** Size of a sample population. */
  private int population_size;
  
  /** Estimated mean vector of standardized variables. */
  private double[] standardized_mean_vector;
  
  /** Estimated variance vector of variables. */
  private double[] variance_vector;
  
  /** Matrix holding correlations between variables. */
  private double[][] correlation_matrix;
  
  /** Dimension of a matrix and vector. */
  private int dimension;

  
  /**
   * Calculates covariance matrix and mean vector given
   * a population of EvRealVector individuals.
   * 
   * @param sample_population sample population
   */
  public void calculateCovarianceMatrix(
      final EvPopulation<EvRealVectorIndividual> sample_population) {
        
    population = sample_population;
    population_size = population.size();
    dimension = population.get(0).getDimension();
    standardized_mean_vector = new double[dimension];
    variance_vector = new double[dimension];
    correlation_matrix = new double[dimension][];
    mean_vector = new double[dimension];
    covariance_matrix = new double[dimension][];
    double lambda_value;
        
    for (int i = 0; i < dimension; i++) {
      covariance_matrix[i] = new double[dimension];
      correlation_matrix[i] = new double[dimension];
    }
    
    for (int i = 0; i < dimension; i++) {
      mean_vector[i] = countMean(i, population);
      variance_vector[i] = countVariance(i, mean_vector[i], population);
      
      if (variance_vector[i] != 0) {
        standardized_mean_vector[i] =
          mean_vector[i] / Math.sqrt(variance_vector[i]);
      } else {
        standardized_mean_vector[i] = mean_vector[i];      
      }
    }
    
    for (int i = 0; i < dimension; i++) {
      for (int j = 0; j < dimension; j++) {
        correlation_matrix[i][j] = countCorrelation(i,
            standardized_mean_vector[i], Math.sqrt(variance_vector[i]), 
            j, standardized_mean_vector[j], Math.sqrt(variance_vector[j]),
            population);     
      }
    }
    
    /* calculate values for covariance matrix */
    for (int i = 0; i < dimension; i++) {
      covariance_matrix[i][i] = variance_vector[i];
    }
    
    lambda_value = countLambdaValue();
    
    for (int i = 0; i < dimension; i++) {
      for (int j = 0; j < dimension; j++) {
        if (i != j) {
          covariance_matrix[i][j] = correlation_matrix[i][j]
            * Math.min(1, Math.max(0, 1 - lambda_value))
            * Math.sqrt(variance_vector[i] * variance_vector[j]);
        }
      }
    }
  }
  
  
  /**
   * Gets estimated covariance matrix.
   * CalculateCovarianceMatrix function should be called first.
   * 
   * @return covariance matrix
   */
  public double[][] getEstimatedCovarianceMatrix() {
    return covariance_matrix;
  }
  
  
  /**
   * Gets estimated mean vector.
   * CalculateCovarianceMatrix function should be called first.
   * 
   * @return mean vector
   */
  public double[] getMeanVector() {
    return mean_vector;
  }
  
  /**
   * Estimates mean of a variable at specified index.
   * 
   * @param index index of a variable
   * @param sample_population population
   * @return mean
   */
  private double countMean(final int index,
      final EvPopulation<EvRealVectorIndividual> sample_population) {
    
    if (index >= sample_population.get(0).getDimension()) {
      throw new IllegalArgumentException("Index is wrong");
    }
    
    double sum = 0;
    
    for (int i = 0; i < sample_population.size(); i++) {
      sum += sample_population.get(i).getValue(index);
    }
    
    return sum / population_size; 
  }  
  
  
  /**
   * Estimates variance for a variable at specified index.
   * 
   * @param index index of a variable
   * @param avg average value (mean) for a variable
   * @param sample_population population
   * @return variance
   */
  private double countVariance(final int index, final double avg,
      final EvPopulation<EvRealVectorIndividual> sample_population) {
    
    if (index >= sample_population.get(0).getDimension()) {
      throw new IllegalArgumentException("Index is wrong");
    }
    
    double sum = 0;
    
    for (int i = 0; i < population_size; i++) {
      sum += Math.pow(sample_population.get(i).getValue(index) - avg, 2);
    }
    
    return sum / (population_size - 1);
  }
  
  
  /**
   * Estimates correlation between two variables.
   * 
   * @param index1 index of the first variable
   * @param avg1 average value of the first variable (mean)
   * @param st_dev1 standard deviation of the first variable
   * @param index2 index of the second variable
   * @param avg2 average value of the second variable (mean)
   * @param st_dev2 standard deviation of the second variable
   * @param sample_population sample population
   * @return estimated correlation
   */
  private double countCorrelation(final int index1, final double avg1,
      final double st_dev1, final int index2, final double avg2,
      final double st_dev2,
      final EvPopulation<EvRealVectorIndividual> sample_population) {
    
    if (index1 >= sample_population.get(0).getDimension()
        || index2 >= sample_population.get(0).getDimension()) {
      throw new IllegalArgumentException("Index is wrong");
    }
        
    double sum = 0;    
    
    for (int i = 0; i < population_size; i++) {
      if (st_dev1 != 0 && st_dev2 != 0) {
        sum += (sample_population.get(i).getValue(index1) / st_dev1 - avg1)
          * (sample_population.get(i).getValue(index2) / st_dev2 - avg2);
      }
    }
    
    return sum / (population_size - 1);
  }
  
  
  /**
   * Calculates auxilary value needed for estimating
   * covariance matrix. Indexes are of two different variables.
   * 
   * @param index1 index of the first variable
   * @param index2 index of the second variable
   * @return returns calculated value
   */
  private double countVarValue(final int index1, final int index2) {
    
    double sum = 0;
    
    for (int i = 0; i < population.size(); i++) {
      if (variance_vector[index1] != 0 && variance_vector[index2] != 0) {
        sum += Math.pow((population.get(i).getValue(index1)
            / Math.sqrt(variance_vector[index1])
            - standardized_mean_vector[index1])
            * (population.get(i).getValue(index2)
            / Math.sqrt(variance_vector[index2])
            - standardized_mean_vector[index2])
            - correlation_matrix[index1][index2], 2);
      }
    }
    
    // off MagicNumber
    return (population_size / Math.pow(population_size - 1, 3)) * sum;
    // on MagicNumber
  }
  
  
  /**
   * Calculates auxilary value needed for estimating
   * covariance matrix.
   * 
   * @return value
   */
  private double countLambdaValue() {
    
    double sum1, sum2;
    
    sum1 = 0;
    sum2 = 0;
    
    for (int i = 0; i < dimension; i++) {
      for (int j = 0; j < dimension; j++) {
        if (i != j) {
          sum1 += countVarValue(i, j);
          sum2 += Math.pow(correlation_matrix[i][j], 2);
        }
      }
    }
    if (sum2 != 0) {
      return sum1 / sum2;
    } else {
      return 0;
    }    
  }
}
