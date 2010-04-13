package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
//import pl.wroc.uni.ii.evolution.utils.EvRandomizer;


/**
 * Linear Bayesian Network used in MIMIC algorithm.
 * 
 * EXPERIMENTAL - USE AT OWN RISK !!
 * 
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 */
public class EvDiscreteVectorMIMICBayesianNetwork {

  /**
   * Dimension of individual.
   */
  private final int dimension;

  /**
   * Table of possible values.
   */
  private int[] possibleValues;
  
  /**
   * Permutation of variables.
   */
  private int[] permutation = null;

  /**
   * Vector of probabilities P(X=1|Y=1).
   */
  //private double[][] probabilities1;

  /**
   * Vector of probabilities P(X=1|Y=0).
   */
  //private double[][] probabilities0;

  /**
   * Matrix of probabilities.
   */
  private double[][][] probabilities;

  /**
   * @param dim dimension of individual
   * @param gVal table of possible gene values
   */
  public EvDiscreteVectorMIMICBayesianNetwork(final int dim, final int[] gVal) {
    dimension = dim;
    possibleValues = gVal;
  }

  

  /**
   * @return new individual based on bayesian network and probabilities
   */
  public EvKnaryIndividual generateIndividual() {

    EvKnaryIndividual individual = new EvKnaryIndividual(dimension, 
        Integer.MAX_VALUE);
    
    // nalezy dopisac losowanie genow dla nowego individuala
    //
    /*
    if (permutation == null) {
      for (int i = 0; i < dimension; i++) {
        individual.setGene(i, EvRandomizer.INSTANCE.nextInt(0, 1, true));
      }
    } else {
      for (int i = dimension - 1; i >= 0; i--) {
        individual.setGene(i, EvRandomizer.INSTANCE
            .nextProbableBooleanAsInt(probabilities1[i]));
      }
    }
    */
    return individual;
  }


  /**
   * @param variables table of variables
   * @return index of the smallest variable in table of variables
   */
  private static int findIndexOfMinValue(final double[] variables) {
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
      //probabilities1 = new double[dimension];
      //probabilities0 = new double[dimension];
      probabilities = new double[possibleValues.length]
                                 [dimension][possibleValues.length];
      
    
    boolean[] isSelected = new boolean[dimension];
    for (int j = 0; j < dimension; j++) {
      isSelected[j] = false;
    }
    double[] entropy = new double[dimension];
    int minEntropy = -1;

    
    
    for (int i = dimension - 1; i >= 0; i--) {
      for (int j = 0; j < dimension; j++) {

        if (i == dimension - 1) {
          // finding first variable of the smallest entropy
          entropy[j] = calculateEntropy(j, population);
        } else {
          // finding permutation of the rest variables

          if (!isSelected[j]) {
            entropy[j] = calculateEntropy(j, minEntropy, population);

          } else {
            entropy[j] = Double.MAX_VALUE;
          }
        }
      }
      
      minEntropy = findIndexOfMinValue(entropy);
      permutation[i] = minEntropy;
      isSelected[minEntropy] = true;
      
      
      if (i == (dimension - 1)) {
       for (int k = 0; k < probabilities.length; k++) {
         for (int l = 0; l < probabilities.length; l++) {
           probabilities[k][i][l] = calculateProbability(permutation[i], k, 
               population);
         }
           
       }
        
      //probabilities1[i] = calculateProbability(permutation[i], 1, population);
      //probabilities0[i] = calculateProbability(permutation[i], 1, population);


      } else {
        
        for (int k = 0; k < probabilities.length; k++) {
          for (int l = 0; l < probabilities.length; l++) {
            probabilities[k][i][l] = calculateProbability(permutation[i], k, 
                permutation[i + 1], l, population);
          }
        }
          
        /*probabilities1[i] =
            calculateProbability(permutation[i], 1, permutation[i + 1], 1,
                population);
        probabilities0[i] =
            calculateProbability(permutation[i], 1, permutation[i + 1], 0,
                population);
        */
      }
    }
  }
    
    // it can be optimalized
  }


  /**
   * Calculates entropy h(X).
   * 
   * @param x index of variable X
   * @param population current population
   * @return entropy h(X)
   */
  public double calculateEntropy(final int x,
      final EvPopulation<EvKnaryIndividual> population) {

    double entropy = 0;
    int x_val;
    double prob;
    
    for (int i = 0; i < possibleValues.length; i++) {
      x_val = possibleValues[i];
      prob = calculateProbability(x, x_val, population);
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
  public double calculateEntropy(final int x, final int y,
      final EvPopulation<EvKnaryIndividual> population) {

    double entropy = 0;
    int x_val;
    int y_val;
    double prob;

    for (int j = 0; j < possibleValues.length; j++) {
      y_val = possibleValues[j];
      for (int i = 0; i < possibleValues.length; i++) {
        x_val = possibleValues[i];
        prob = calculateProbability(x, x_val, y, y_val, population);
        if (prob != 0) {
          entropy -= (prob * Math.log(prob));
        }
      }
    }

    return entropy;
  }


  /**
   * Calculate probability p(x=x_val|y=y_val).
   * 
   * @param x index of variable Y
   * @param x_val value of x-th variable
   * @param y index of variable Y
   * @param y_val value of y-th variable
   * @param population current population
   * @return probability p(x=x_val|y=y_val)
   */
  public double calculateProbability(final int x, final int x_val,
      final int y, final int y_val,
      final EvPopulation<EvKnaryIndividual> population) {

    double prob = 0;
    for (EvKnaryIndividual ind : population) {

      if ((ind.getGene(x) == x_val) && (ind.getGene(y) == y_val)) {
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
  public double calculateProbability(final int x, final int x_val,
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
   * Gets probability P(X=1|Y=0) table.
   * @return table of probability P(X=1|Y=0)
   */
  /*public double[] getProbabilitiesZero() {
    return this.probabilities0;
  }*/
  
  /**
   * Gets probability P(X=1|Y=1) table.
   * @return table of probability P(X=1|Y=1)
   */
  /*public double[] getProbabilitiesOne() {
    return this.probabilities1;
  }*/
  
  
  /**
   * {@inheritDoc}
   */
  public String toString() {
    String s = new String();
    
    s += "Network permutation:\n { ";
    for (int i = 1; i <= dimension; i++) {
      s += permutation[i - 1];
      if (i < dimension) {
        s += ", ";
      }
    }
    s += " } ;\n";
    
    s += "Network probabilities:\n ";
    for (int i = 0; i < probabilities.length; i++) {
      for (int j = 0; j < probabilities[i].length; j++) {
        s = s + "[" + i + "," + j + "]\n";
        for (int k = 0; k < probabilities[i][j].length; k++) {
          s += " " + probabilities[i][j][k] + " ";
        }
        s += "\n";
      }
    }
    
    
    return s;
  }

}
