package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Linear Bayesian Network used in MIMIC algorithm - binary version.
 * 
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 */
public class EvBinaryVectorMIMICBayesianNetwork {

  
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
   * Size of a population.
   */
  private int population_size;
  
  /**
   * @return population_size
   */
  protected int getPopulationSize() {
    return population_size;
  }
  
  /**
   * @param p population_size
   */
  protected void setPopulationSize(final int p) {
    population_size = p;
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
   * Vector of probabilities P(X=1|Y=1).
   */
  private double[] probabilities1;
  
  /**
   * @return probabilities1
   */
  protected double[] getProbabilities1() {
    return probabilities1;
  }
  
  /**
   * @param p probabilities1
   */
  protected void setProbabilities1(final double[] p) {
    probabilities1 = p;
  }
  

  /**
   * Vector of probabilities P(X=1|Y=0).
   */
  private double[] probabilities0;
  
  /**
   * @return probabilities0
   */
  protected double[] getProbabilities0() {
    return probabilities0;
  }
  
  /**
   * @param p probabilities0
   */
  protected void setProbabilities0(final double[] p) {
    probabilities1 = p;
  }
  
  
  /**
   * Objective function.
   */
  private 
  EvObjectiveFunction<EvBinaryVectorIndividual> objective_function;
  
  /**
   * @return objective_function
   */
  protected EvObjectiveFunction<EvBinaryVectorIndividual> 
      getObjectiveFunction() {
    return objective_function;
  }
  
  /**
   * @param f objective_function
   */
  protected void setObjectiveFunction(final 
      EvObjectiveFunction<EvBinaryVectorIndividual> f) {
    objective_function = f;
  }
  

  /**
   * Constructor.
   * @param dim dimension of individual
   * @param function objective function
   */
  public EvBinaryVectorMIMICBayesianNetwork(final int dim, 
      final EvObjectiveFunction<EvBinaryVectorIndividual> function) {
    dimension = dim;
    objective_function = function;
  }


  /**
   * @return new individual based on bayesian network and probabilities
   */
  public EvBinaryVectorIndividual generateIndividual() {
    
    EvBinaryVectorIndividual individual =
        new EvBinaryVectorIndividual(dimension);
    
    boolean previous = false;
    boolean current = false;

    if (permutation == null) {
      for (int i = 0; i < dimension; i++) {
        individual.setGene(i, EvRandomizer.INSTANCE.nextInt(0, 1, true));
      }
      
    } else {
      for (int i = dimension - 1; i >= 0; i--) {
        
        if (previous) {
          current = 
            EvRandomizer.INSTANCE.nextProbableBoolean(
                probabilities1[permutation[i]]);
        } else {
          current = 
            EvRandomizer.INSTANCE.nextProbableBoolean(
                probabilities0[permutation[i]]);
        }
        
        if (current) {
          individual.setGene(permutation[i], 1);
        } else {
          individual.setGene(permutation[i], 0);
        }
        
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
      final EvPopulation<EvBinaryVectorIndividual> population) {

    if (permutation == null) {
      permutation = new int[dimension];
      probabilities1 = new double[dimension];
      probabilities0 = new double[dimension];
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
        double prob = calculateProbability(permutation[i], 1, population);
        probabilities1[permutation[i]] = prob; 
        probabilities0[permutation[i]] = prob;


      } else { // dependent variable
        probabilities1[permutation[i]] =
            calculateConditionalProbability(permutation[i], 1, 
                permutation[i + 1], 1, population);
        probabilities0[permutation[i]] =
            calculateConditionalProbability(permutation[i], 1, 
                permutation[i + 1], 0, population);
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
      final EvPopulation<EvBinaryVectorIndividual> population) {
    
    double entropy = 0;
    for (int x_val = 0; x_val <= 1; x_val++) {
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
      final EvPopulation<EvBinaryVectorIndividual> population) {
    
    double entropy = 0;
    
    for (int y_val = 0; y_val <= 1; y_val++) {
      for (int x_val = 0; x_val <= 1; x_val++) {
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
      final EvPopulation<EvBinaryVectorIndividual> population) {
    
    double count = 0;
    double universum = 0;
    
    for (EvBinaryVectorIndividual ind : population) {
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
      final EvPopulation<EvBinaryVectorIndividual> population) {

    double prob = 0;
    
    for (EvBinaryVectorIndividual ind : population) {
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
      final EvPopulation<EvBinaryVectorIndividual> population) {

    double prob = 0;
    for (EvBinaryVectorIndividual individual : population) {

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
    s += " } ; ";
    
    s += "[ ";
    for (int i = 1; i <= dimension; i++) {
      s += probabilities0[i - 1];
      if (i < dimension) {
        s += ", ";
      }
    }
    s += " ] ; [ ";
    for (int i = 1; i <= dimension; i++) {
      s += (probabilities1[i - 1]);
      if (i < dimension) {
        s += ", ";
      }
    }
    s += " ]";    
    
    return s;
  }

}
