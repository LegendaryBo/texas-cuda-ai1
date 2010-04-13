package pl.wroc.uni.ii.evolution.engine.operators.general.selections.multiobjective;

import static org.junit.Assert.assertTrue;
import org.junit.Test;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvBinaryPattern;

/**  @author Adam Palka */
public class EvParetoFrontsRankTest {

  //off MagicNumbers
  
  /**  Test for two objective functions and EvBinaryVectorIndividual. */
  @Test
  public void testForBinaryPattern2ObjectiveFunctions() {
    EvPopulation<EvBinaryVectorIndividual> population =
      new EvPopulation<EvBinaryVectorIndividual>();    
    population.add(
        new EvBinaryVectorIndividual(new int[] {1, 1, 1, 1}));
    population.add(
        new EvBinaryVectorIndividual(new int[] {1, 1, 0, 1}));
    population.add(
        new EvBinaryVectorIndividual(new int[] {1, 1, 1, 0}));
    population.add(
        new EvBinaryVectorIndividual(new int[] {0, 0, 0, 0}));
    population.add(
        new EvBinaryVectorIndividual(new int[] {0, 0, 1, 1}));
    population.add(
        new EvBinaryVectorIndividual(new int[] {0, 0, 0, 0}));
    for (EvBinaryVectorIndividual b : population) {
      b.addObjectiveFunction(
          new EvBinaryPattern(new int[] {1, 0, 1, 1}));
      b.addObjectiveFunction(
          new EvBinaryPattern(new int[] {0, 1, 1, 1}));
    }   
    EvParetoFrontsRank<EvBinaryVectorIndividual> rank =
      new EvParetoFrontsRank<EvBinaryVectorIndividual>(population);
    
    assertTrue("Individuals 0 and 4 should have the same rank",
        rank.getRank(0) == rank.getRank(4));
    assertTrue("Individuals 1 and 2 should have the same rank",
        rank.getRank(1) == rank.getRank(2));
    assertTrue("Individuals 3 and 5 should have the same rank",
        rank.getRank(3) == rank.getRank(5));    
    assertTrue("Individual 0 should have bigger rank than individual 1",
        rank.getRank(0) > rank.getRank(1));
    assertTrue("Individual 0 should have bigger rank than individual 3",
        rank.getRank(0) > rank.getRank(3));
    assertTrue("Individual 1 should have bigger rank than individual 3",
        rank.getRank(1) > rank.getRank(3));  
  }
  
  /** Test for three objective functions and EvBinaryVectorIndividual. */
  @Test
  public void testForBinaryPattern3ObjectiveFunctions() {
    EvPopulation<EvBinaryVectorIndividual> population =
      new EvPopulation<EvBinaryVectorIndividual>();    
    population.add(new EvBinaryVectorIndividual(
        new int[] {0, 1, 0, 1, 0, 1, 0, 1}));
    population.add(new EvBinaryVectorIndividual(
        new int[] {0, 1, 1, 1, 0, 1, 1, 1}));
    population.add(new EvBinaryVectorIndividual(
        new int[] {0, 1, 0, 1, 0, 1, 0, 1}));
    population.add(new EvBinaryVectorIndividual(
        new int[] {0, 0, 1, 1, 0, 0, 1, 1}));
    population.add(new EvBinaryVectorIndividual(
        new int[] {1, 0, 1, 1, 1, 0, 1, 1}));
    population.add(new EvBinaryVectorIndividual(
        new int[] {0, 0, 0, 0, 1, 1, 1, 1}));
    for (EvBinaryVectorIndividual b : population) {
      b.addObjectiveFunction(new EvBinaryPattern(
          new int[] {0, 1, 1, 1, 1, 1, 1, 0}));
      b.addObjectiveFunction(new EvBinaryPattern(
          new int[] {0, 0, 1, 1, 0, 0, 1, 1}));
      b.addObjectiveFunction(new EvBinaryPattern(
          new int[] {0, 0, 0, 0, 1, 1, 1, 1}));
    }   
    EvParetoFrontsRank<EvBinaryVectorIndividual> rank =
      new EvParetoFrontsRank<EvBinaryVectorIndividual>(population);
    
    assertTrue("Individuals 0 and 2 should have the same rank",
        rank.getRank(0) == rank.getRank(2));
    assertTrue("Individuals 1 and 3 should have the same rank",
        rank.getRank(1) == rank.getRank(3));
    assertTrue("Individuals 1 and 5 should have the same rank",
        rank.getRank(1) == rank.getRank(5));  
    assertTrue("Individual 1 should have bigger rank than individual 0",
        rank.getRank(1) > rank.getRank(0));
    assertTrue("Individual 1 should have bigger rank than individual 4",
        rank.getRank(3) > rank.getRank(4));
    assertTrue("Individual 0 should have smaller rank than individual 4",
        rank.getRank(0) < rank.getRank(4));
  }
  
  /**
   * Test if computed rank is ok.
   * @author Mateusz Poslednik mateusz.poslednik@gmail.com
   */
  @Test
  public void getRankTest() {
    EvObjectiveFunctionTest obja = 
      new EvObjectiveFunctionTest<EvRealVectorIndividual>(0);
    EvObjectiveFunctionTest objb = 
    new EvObjectiveFunctionTest<EvRealVectorIndividual>(1);
    
    double[] gene1 = {3, 8};
    EvRealVectorIndividual in1 = new EvRealVectorIndividual(gene1);
    in1.addObjectiveFunction(obja);
    in1.addObjectiveFunction(objb);
    double[] gene2 = {2, 6};
    EvRealVectorIndividual in2 = new EvRealVectorIndividual(gene2);
    in2.addObjectiveFunction(obja);
    in2.addObjectiveFunction(objb);
    double[] gene3 = {0, 3};
    EvRealVectorIndividual in3 = new EvRealVectorIndividual(gene3);
    in3.addObjectiveFunction(obja);
    in3.addObjectiveFunction(objb);
    double[] gene4 = {4, 2};
    EvRealVectorIndividual in4 = new EvRealVectorIndividual(gene4);
    in4.addObjectiveFunction(obja);
    in4.addObjectiveFunction(objb);
    double[] gene5 = {6, 4};
    EvRealVectorIndividual in5 = new EvRealVectorIndividual(gene5);
    in5.addObjectiveFunction(obja);
    in5.addObjectiveFunction(objb);
    double[] gene6 = {8, 5};
    EvRealVectorIndividual in6 = new EvRealVectorIndividual(gene6);
    in6.addObjectiveFunction(obja);
    in6.addObjectiveFunction(objb);
    double[] gene7 = {9, 1};
    EvRealVectorIndividual in7 = new EvRealVectorIndividual(gene7);
    in7.addObjectiveFunction(obja);
    in7.addObjectiveFunction(objb);
    double[] gene8 = {10, 0};
    EvRealVectorIndividual in8 = new EvRealVectorIndividual(gene8);
    in8.addObjectiveFunction(obja);
    in8.addObjectiveFunction(objb);
    
    EvPopulation<EvRealVectorIndividual> population =
      new EvPopulation<EvRealVectorIndividual>();
    population.add(in1);
    population.add(in2);
    population.add(in3);
    population.add(in4);
    population.add(in5);
    population.add(in6);
    population.add(in7);
    population.add(in8);
    
    EvParetoFrontsRank<EvRealVectorIndividual> rank =
      new EvParetoFrontsRank<EvRealVectorIndividual>(population);
    
    assertTrue("Rank 1 and 6 should be equal.", 
        rank.getRank(0) == rank.getRank(5));
    assertTrue("Rank 1 should be bigger than 5.", 
        rank.getRank(0) > rank.getRank(4));
    assertTrue("Rank 2 should be bigger than 4.", 
        rank.getRank(1) > rank.getRank(3));
    assertTrue("Rank 2 and 5 should be equal.", 
        rank.getRank(1) == rank.getRank(4));
    assertTrue("Rank 7 should be bigger than 4.", 
        rank.getRank(3) < rank.getRank(6));
    assertTrue("Rank 7 should be bigger than 2.", 
        rank.getRank(1) < rank.getRank(6));
    assertTrue("Rank 1 and 8 should be equal.", 
        rank.getRank(0) == rank.getRank(7));
  }
  
  /**
   * Temporary object to help make easier testing.
   * @author Mateusz Poslednik mateusz.poslednik@gmail.com
   *
   * @param <T> Type of individual to be evaluated.
   */
  class EvObjectiveFunctionTest<T extends EvRealVectorIndividual> 
      implements EvObjectiveFunction<T> {

    /** index of gene to be returned. */
    private int index;
    
    /**
     * Default constructor.
     * @param index_ Index of gene.
     */
    public EvObjectiveFunctionTest(final int index_) {
      this.index = index_;
    }
    
    /**
     * Return value of objective function for the individual.
     * @param individual Individual to compute his obj. fun.
     * @return objective function value.
     */
    public double evaluate(final T individual) {
      return individual.getValue(index);
    }
    
  }
  //on MagicNumbers
  
}
