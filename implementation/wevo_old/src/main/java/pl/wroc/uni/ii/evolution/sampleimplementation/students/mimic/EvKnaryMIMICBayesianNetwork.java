package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Linear Bayesian Network used in MIMIC algorithm - k-nary version.
 * 
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 */
public class EvKnaryMIMICBayesianNetwork {

  
  /**
   * Dimension of individual.
   */
  private int dimension;
  
  /**
   * @return dimension
   */
  protected int getDimension() {
    return dimension;
  }
  
  /**
   * @param d dimension
   */
  protected void setDimension(final int d) {
    dimension = d;
  }

  
  /**
   * Number of values which variable can assume.
   * Set of values: {0, 1, ... numberOfValues-1}
   */
  private int numberOfValues;
  
  /**
   * @return numberOfValues
   */
  protected int getNumberOfValues() {
    return numberOfValues;
  }
  
  /**
   * @param n numberOfValues
   */
  protected void setNumberOfValues(final int n) {
    numberOfValues = n;
  }

  
  /**
   * Contains indexes of variables.
   * Determine permutation of variables.
   * Direction of relation:
   * permutation[0] <- permutation[1] <- ... <- permutation[n]
   * Only the last variable is independent.  
   */
  private int[] permutation = null;
  
  /**
   * @return permutation
   */
  protected int[] getPermutation() {
    return permutation;
  }
  
  /**
   * @param p permutation
   */
  protected void setPermutation(final int[] p) {
    permutation = p;
  }
  
  
  /**
   * Vector of probabilities.
   * probability[i][y][x] = P(X[i]=x|X[j]=y)
   * where if permutation[i] = k, then permutation[i+1] = j
   */
  private double[][][] probabilities;
  
  /**
   * @return probabilities
   */
  protected double[][][] getProbabilities() {
    return probabilities;
  }
  
  /**
   * @param p probabilities
   */
  protected void setProbabilities(final double[][][] p) {
    probabilities = p;
  }
  
  
  /**
   * Objective function.
   */
  private 
  EvObjectiveFunction<EvKnaryIndividual> objective_function;
  
  /**
   * @return objective_function
   */
  protected EvObjectiveFunction<EvKnaryIndividual> 
      getObjectiveFunction() {
    return objective_function;
  }
  
  /**
   * @param f objective_function
   */
  protected void setObjectiveFunction(final 
      EvObjectiveFunction<EvKnaryIndividual> f) {
    objective_function = f;
  }
  

  /**
   * Constructor.
   * @param dim dimension of individual
   * @param values number of possible values
   * @param function objective function
   */
  public EvKnaryMIMICBayesianNetwork(final int dim, final int values, 
      final EvObjectiveFunction<EvKnaryIndividual> function) {
    dimension = dim;
    objective_function = function;
    numberOfValues = values;
  }

  
  /**
   * Returns value from [0,...,numberOfValues].
   * (in accordance with probability in probabilities table) 
   * @param pv value of previous variable
   * @param i index of current variable
   * @return random value
   */
  protected int random(final int pv, final int i) {
    
    double rand = EvRandomizerExtended.INSTANCE.nextDouble();
    int value = 0;
    double sum = probabilities[i][pv][value];
    while (sum < rand) {
      value++;
      sum += probabilities[i][pv][value];
    }
    return value;
  }
  

  /**
   * @return new individual based on bayesian network and probabilities
   */
  public EvKnaryIndividual generateIndividual() {
    
    EvKnaryIndividual individual =
        new EvKnaryIndividual(dimension, numberOfValues - 1);
    
    int previous = 0;
    int current = 0;

    if (permutation == null) {
      for (int i = 0; i < dimension; i++) {
        individual.setGene(i, EvRandomizer.INSTANCE.nextInt(0, 
            numberOfValues - 1, true));
      }
      
    } else {
      for (int i = dimension - 1; i >= 0; i--) {
        
        current = random(previous, permutation[i]);
        individual.setGene(permutation[i], current);
        previous = current;   
      }
    }
    
    individual.setObjectiveFunction(objective_function);
    return individual;
  }


  /**
   * @param variables table of variables
   * @return index of the smallest variable in table of variables
   */
  protected static int findIndexOfMinValue(final double[] variables) {
    int index = 0;
    double value = variables[0];
    for (int i = 1; i < variables.length; i++) {
      if (variables[i] < value) {
        index = i;
        value = variables[i];
      }
    }
    return index;
  }


  /**
   * Greedy algorithm which is finding optimal permutation of variables.
   * Estimates probabilities (using current population).
   * 
   * @param population current population
   */
  public void estimateProbabilities(
      final EvPopulation<EvKnaryIndividual> population) {

    if (permutation == null) {
      permutation = new int[dimension];
      probabilities = new double[dimension][numberOfValues][numberOfValues];
    }
    
    boolean[] isSelected = new boolean[dimension];
    for (int j = 0; j < dimension; j++) {
      isSelected[j] = false;
    }
    
    double[] entropy = new double[dimension];
    int minEntropyIndex = -1;

    for (int i = dimension - 1; i >= 0; i--) {
      for (int j = 0; j < dimension; j++) {

        if (i == dimension - 1) {
          // finding first variable of the smallest entropy
          entropy[j] = calculateEntropy(j, population);
          
        } else {
          // finding permutation of the rest variables
          if (!isSelected[j]) {
            entropy[j] = calculateEntropy(j, minEntropyIndex, population);

          } else {
            entropy[j] = Double.MAX_VALUE;
          }
        }
      }
      
      minEntropyIndex = findIndexOfMinValue(entropy);
      permutation[i] = minEntropyIndex;
      isSelected[minEntropyIndex] = true;
      
      if (i == (dimension - 1)) { // independent variable
        
        for (int x = 0; x < numberOfValues; x++) {
          double prob = calculateProbability(permutation[i], x, population);
          for (int y = 0; y < numberOfValues; y++) {
            probabilities[permutation[i]][y][x] = prob;
          }
        }

      } else { // dependent variable
        
        for (int x = 0; x < numberOfValues; x++) {
          for (int y = 0; y < numberOfValues; y++) {
            probabilities[permutation[i]][y][x] = calculateProbability(
                permutation[i], x, permutation[i + 1], y, population);
          }
        }
      }
    }
    probabilitiesCorrection();
  }
  
  
  // off MagicNumber
  /**
   * Permissible error in sum of probabilities.
   * (sum should be equal 1) 
   */
  private double probabilityPermissibleError = 0.001;
  // on MagicNumber
  
  /**
   * @return probabilityPermissibleError
   */
  protected double getProbabilityPermissibleError() {
    return probabilityPermissibleError;
  }
  
  /**
   * @param e probabilityPermissibleError
   */
  protected void setProbabilityPermissibleError(final double e) {
    probabilityPermissibleError = e;
  }
  
  
  /**
   * It finds defects in table of probabilities.
   * It may happened, if in population will be no individual, which
   * have on i-th gene value y.
   * Than in table will be P(...|X(i)=y) and sum of probabilities
   * will be not 1.
   * In this situation algorithm assign the same probability
   * for every possible value.
   */
  protected void probabilitiesCorrection() {
    
    for (int i = 0; i < dimension; i++) {
      for (int y = 0; y < numberOfValues; y++) {
        double sum = 0;
        for (int x = 0; x < numberOfValues; x++) {
          sum += probabilities[i][y][x];
        }
        
        if ((sum > (1.0 + probabilityPermissibleError))
            || (sum < (1.0 - probabilityPermissibleError))) {
          
          double prob = 1.0 / numberOfValues; 
          for (int x = 0; x < numberOfValues; x++) {
            probabilities[i][y][x] = prob;
          }
        }
      }
    }
  }


  /**
   * Calculates entropy h(X).
   * 
   * @param x index of variable X
   * @param population current population
   * @return entropy h(X)
   */
  protected double calculateEntropy(final int x,
      final EvPopulation<EvKnaryIndividual> population) {
    
    double entropy = 0;
    for (int x_val = 0; x_val < numberOfValues; x_val++) {
      double prob = calculateProbability(x, x_val, population);
      if (prob != 0) {
        entropy -= (prob * Math.log(prob));
      }
    }
    return entropy;
  }


  /**
   * Calculates entropy h(X|Y).
   * 
   * @param x index of variable X
   * @param y index of variable Y
   * @param population current population
   * @return entropy h(X|Y)
   */
  protected double calculateEntropy(final int x, final int y,
      final EvPopulation<EvKnaryIndividual> population) {
    
    double entropy = 0;
    
    for (int y_val = 0; y_val < numberOfValues; y_val++) {
      for (int x_val = 0; x_val < numberOfValues; x_val++) {
        double prob_x_y = calculateProbability(x, x_val, y, y_val, population);
        double prob_y = calculateProbability(y, y_val, population);
        
        if (prob_x_y != 0) {
          entropy -= (prob_x_y * Math.log(prob_x_y / prob_y));
        }
      }
    }
    return entropy;
  }

  
  /**
   * Calculate probability p (x=x_val | y=y_val).
   * 
   * @param x index of variable X
   * @param x_val value of x-th variable
   * @param y index of variable Y
   * @param y_val value of y-th variable
   * @param population current population
   * @return probability p ( x=x_val | y=y_val)
   */
  protected double calculateConditionalProbability(final int x, 
      final int x_val, final int y, final int y_val, 
      final EvPopulation<EvKnaryIndividual> population) {
    
    double count = 0;
    double universum = 0;
    
    for (EvKnaryIndividual ind : population) {
      if (ind.getGene(y) == y_val) {
        universum++;
        if (ind.getGene(x) == x_val) {
          count++;
        }
      }
    }
    
    if (universum == 0) {
      return 0.0;
    } else {
      return count / universum;
    }
  }

  /**
   * Calculate probability p (x=x_val & y=y_val).
   * 
   * @param x index of variable X
   * @param x_val value of x-th variable
   * @param y index of variable Y
   * @param y_val value of y-th variable
   * @param population current population
   * @return probability p (x=x_val & y=y_val)
   */
  protected double calculateProbability(final int x, final int x_val,
      final int y, final int y_val,
      final EvPopulation<EvKnaryIndividual> population) {

    double prob = 0;
    
    for (EvKnaryIndividual ind : population) {
      if ((ind.getGene(y) == y_val) && (ind.getGene(x) == x_val)) {
          prob++;
      }
    }
    prob /= population.size();
    return prob;
  }


  /**
   * Calculate probability p(x=x_val).
   * 
   * @param x index of variable X
   * @param x_val value of x-th variable
   * @param population current population
   * @return probability p(x=x_val)
   */
  protected double calculateProbability(final int x, final int x_val,
      final EvPopulation<EvKnaryIndividual> population) {

    double prob = 0;
    for (EvKnaryIndividual individual : population) {

      if (individual.getGene(x) == x_val) {
        prob++;
      }
    }
    prob /= population.size();
    return prob;
  }
  
    
  /**
   * {@inheritDoc}
   */
  public String toString() {
    String s = new String();
    
    s += "{ ";
    for (int i = 1; i <= dimension; i++) {
      s += permutation[i - 1];
      if (i < dimension) {
        s += ", ";
      }
    }
    s += " }\n";
    
    for (int i = 0; i < dimension; i++) {
      s += " " + i + ")\n";
      for (int j = 1; j <= numberOfValues; j++) {
        s += "     [ ";
        for (int k = 1; k <= numberOfValues; k++) {
          s += probabilities[i][j][k];
          if (k < numberOfValues) {
            s += ", ";
          }
        }
        s += ("] | " + j + "\n");  
      }
    }
    
    return s;
  }

}
