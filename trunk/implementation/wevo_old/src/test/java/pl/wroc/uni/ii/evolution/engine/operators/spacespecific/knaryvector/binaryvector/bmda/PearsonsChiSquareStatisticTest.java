package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda;

import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.EvPearsonsChiSquareStatistic;

/**
 * Test Pearson's chi-square statistic for binary individuals.
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 *
 */
public class PearsonsChiSquareStatisticTest {
  //off VisibilityModifier
  /** Population to be tested. */
  private EvPopulation<EvBinaryVectorIndividual> population;
  //on VisibilityModifier
  //off MagicNumbers
  
  /**
   * Setting up variables.
   * @throws Exception exception
   */
  @Before
  public void setUp() throws Exception {
    int[] gin1 = {1, 0, 1, 1};
    int[] gin2 = {1, 0, 0, 1};
    int[] gin3 = {0, 1, 0, 1};
    int[] gin4 = {0, 1, 0, 1};
    int[] gin5 = {0, 1, 1, 1};
    int[] gin6 = {1, 0, 1, 1};
    int[] gin7 = {1, 0, 0, 1};
    EvBinaryVectorIndividual in1 = new EvBinaryVectorIndividual(gin1);
    EvBinaryVectorIndividual in2 = new EvBinaryVectorIndividual(gin2);
    EvBinaryVectorIndividual in3 = new EvBinaryVectorIndividual(gin3);
    EvBinaryVectorIndividual in4 = new EvBinaryVectorIndividual(gin4);
    EvBinaryVectorIndividual in5 = new EvBinaryVectorIndividual(gin5);
    EvBinaryVectorIndividual in6 = new EvBinaryVectorIndividual(gin6);
    EvBinaryVectorIndividual in7 = new EvBinaryVectorIndividual(gin7);
    population = new EvPopulation<EvBinaryVectorIndividual>();
    population.add(in1);
    population.add(in2);
    population.add(in3);
    population.add(in4);
    population.add(in5);
    population.add(in6);
    population.add(in7);
  }

  /**
   * Test of computation.
   */
  @Test
  public void testComputeX() {
    EvPearsonsChiSquareStatistic pearson = 
      new EvPearsonsChiSquareStatistic(population);
    try {
      double x1 = pearson.computeX(0, 1);
      double x2 = pearson.computeX(0, 2);
      double x3 = pearson.computeX(2, 3);
      assertTrue("Error betwen gen 0 and 1.", x1 >= 3.84);   //dependent
      assertTrue("Error betwen gen 0 and 2.", x2 < 3.84);   //independent
      assertTrue("Error betwen gen 2 and 3.", x3 < 3.84);   //independent
    } catch (Exception e) {
      e.printStackTrace();
      assertTrue("Failed with exception.", false);
    }
    
  }

  //on MagicNumbers
}
