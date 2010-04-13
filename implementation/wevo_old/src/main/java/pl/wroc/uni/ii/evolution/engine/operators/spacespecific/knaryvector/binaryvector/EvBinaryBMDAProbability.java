package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.bayesian.EvBayesianNetworkStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.EvPearsonsChiSquareStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.bayesnetwork.EvBinaryBayesianNetwork;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvBayesianOperator;

/**
 * Generate new population based on bayesian network. Bayesian network is
 * generated on the selected individuals using Pearson's chi-square statistics.
 * 
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 */
public class EvBinaryBMDAProbability implements
    EvOperator<EvBinaryVectorIndividual>, EvBayesianOperator {
  
  //off MagicNumber
  /** 
   * What level of indepedence are we interested.
   * e.g.: 
   * indepedence = 3.84 then 
   * we want to get depedence on level 95%
   * Default value is 3.84
   */
  private double indepedence = 3.84;
  //on MagicNumber
  /** Size of new population. */
  private final int populationSize;
  /**
   * Max number of parents in bayesian network.
   * Default value is 2.
   * Warning!
   * The bigger value the slower run.
   */
  private int bayesianParents = 2;
  
  /**
   * Storage object in which we store bayesian network statistics.
   */
  private EvPersistentStatisticStorage storage;

  /**
   * Constructor.
   * 
   * @param population_size_ Size of new population
   */
  public EvBinaryBMDAProbability(final int population_size_) {
    this.populationSize = population_size_;
  }


  /**
   * Algorithm:.
   * Generate bayesian network depend on Pearson's chi-square statistics.
   * Generate new population using bayesian network.
   * @param population Selected individuals from the whole population.
   * @return New population - size = this.population_size
   */
  public EvPopulation<EvBinaryVectorIndividual> apply(
      final EvPopulation<EvBinaryVectorIndividual> population) {

    int dimension = population.get(0).getDimension();
    EvBinaryBayesianNetwork net = new EvBinaryBayesianNetwork(bayesianParents);
    net.initialize(population);
    EvPearsonsChiSquareStatistic pearsonTest = 
      new EvPearsonsChiSquareStatistic(population);
    
    for (int i = 0; i < dimension - 1; i++) {
      for (int j = i + 1; j < dimension; j++) {
        try {
          double x = pearsonTest.computeX(i, j);
          if (x > this.indepedence) {
            //we don't know which gen is depend on which one
            net.addEdge(i, j);
            net.addEdge(j, i);
          }
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    }
    EvPopulation<EvBinaryVectorIndividual> newPopulation = 
      new EvPopulation<EvBinaryVectorIndividual>();
    for (int i = 0; i < this.populationSize; i++) {
      newPopulation.add(net.generate());
    }
    
    
    // collect tatistics
    if (storage != null) {
      EvBayesianNetworkStatistic stat = 
        new EvBayesianNetworkStatistic(net.getEdges(), net.getSize());
      storage.saveStatistic(stat);
    }
    
    
    
    return newPopulation;
  }


  /**
   * getter.
   * @return level of indepedence.
   */
  public double getIndepedence() {
    return indepedence;
  }


  /**
   * setter.
   * @param indepedence_ level of indepedence
   */
  public void setIndepedence(final double indepedence_) {
    this.indepedence = indepedence_;
  }

  /**
   * Getter.
   * @return population size
   */
  public int getPopulationSize() {
    return populationSize;
  }

  /**
   * Getter. 
   * @return get max number of parents in bayesian network
   */
  public int getBayesianParents() {
    return bayesianParents;
  }

  /**
   * Setter.
   * @param bayesianParents_ set max number of parents in bayesian network
   */
  public void setBayesianParents(final int bayesianParents_) {
    this.bayesianParents = bayesianParents_;
  }


  /**
   * {@inheritDoc}
   */
  public void collectBayesianStats(
      final EvPersistentStatisticStorage storage_) {
    storage = storage_;    
  }

}
