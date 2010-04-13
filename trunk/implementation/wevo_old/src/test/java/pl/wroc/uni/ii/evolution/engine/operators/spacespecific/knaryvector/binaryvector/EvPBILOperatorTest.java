package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import java.lang.reflect.Field;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorPBILOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * PBIL operator tests.
 * 
 * @author Marcin Golebiowski i Kacper Gorski
 */
public class EvPBILOperatorTest extends TestCase {

  /*
   * In order to test initialProbabilityVector i use java reflection to see
   * values of probablity_vector vector
   * 
   */
  @SuppressWarnings("unchecked")
  public void testInitialProbabilityVector() {

    for (int j : new int[] { 1, 10, 100, 200 }) {
      EvBinaryVectorPBILOperator pbil = new EvBinaryVectorPBILOperator(new EvOneMax(), 1, 0.0, 0.0,
          0.0, j, new EvBinaryVectorSpace(new EvOneMax(), j));
      Class pbil_reflection = pbil.getClass();
      double probability_vector[] = null;
      try {
        Field field_probability_vector = pbil_reflection
            .getDeclaredField("probability_vector");
        field_probability_vector.setAccessible(true);
        probability_vector = (double[]) field_probability_vector.get(pbil);
        assertEquals(probability_vector.length, j);

      } catch (Exception ex) {
        fail(ex.getMessage());
      }

      for (int i = 0; i < probability_vector.length; i++) {
        if (probability_vector[i] != 0.5) {
          fail("Initial probabilty vector is bad");
        }
      }
    }
  }

  /*
   * In order to test correctnes of public function which returns void, i use
   * java reflection
   * 
   */
  public void testSetSolutionSpace() {
    for (int j : new int[] { 0, 1, 10, 3333, 6000, 200000 }) {
      EvBinaryVectorSpace some_solution_space = new EvBinaryVectorSpace(
          new EvOneMax(), j);
      EvBinaryVectorPBILOperator pbil = new EvBinaryVectorPBILOperator(new EvOneMax(), 1, 0.0, 0.0,
          0.0, 30, some_solution_space);
      Class pbil_reflection = pbil.getClass();
      try {
        Field field_solution_space = pbil_reflection
            .getDeclaredField("solution_space");
        field_solution_space.setAccessible(true);
        assertTrue(some_solution_space
            .equals((EvBinaryVectorSpace) field_solution_space.get(pbil)));

      } catch (Exception ex) {
        fail(ex.getMessage());
      }
    }
  }

  /*
   * Depends on names of private fields in PBIL class
   */
  public void testOfParameters() {
    EvBinaryVectorPBILOperator pbil = new EvBinaryVectorPBILOperator(new EvOneMax(), 1, 0.34, 0.45,
        0.65, 3650, new EvBinaryVectorSpace(new EvOneMax(), 3650));
    Class pbil_reflection = pbil.getClass();

    try {

      Field field_theta1 = pbil_reflection.getDeclaredField("theta1");
      field_theta1.setAccessible(true);
      assertEquals((Double) field_theta1.get(pbil), 0.34);

      Field field_theta2 = pbil_reflection.getDeclaredField("theta2");
      field_theta2.setAccessible(true);
      assertEquals((Double) field_theta2.get(pbil), 0.45);

      Field field_theta3 = pbil_reflection.getDeclaredField("theta3");
      field_theta3.setAccessible(true);
      assertEquals((Double) field_theta3.get(pbil), 0.65);

      Field field_n = pbil_reflection.getDeclaredField("n");
      field_n.setAccessible(true);
      assertEquals((Integer) field_n.get(pbil), (Integer) 3650);
    } catch (Exception ex) {
      fail();

    }

  }

  /*
   * In order to test correctnes of PBIL operator we run sample predictable PBIL
   * experiment. In very rare cases this test may fail.
   */
  public void testDoIteration() {
    final int dimension = 10;

    EvBinaryVectorPBILOperator pbil = new EvBinaryVectorPBILOperator(new EvOneMax(), 10, 0.03, 0.02,
        0.02, 100, new EvBinaryVectorSpace(new EvOneMax(), dimension));

    for (int i = 0; i < 10; i++) {
      pbil.apply(null);
    }

    EvBinaryVectorIndividual best_individual_in_pbil = pbil.apply(null)
        .getBestResult();

    EvBinaryVectorIndividual best_individual = new EvBinaryVectorIndividual(dimension);

    for (int i = 0; i < dimension; i++) {
      best_individual.setGene(i, 1);
    }

    for (int i = 0; i < dimension; i++) {
      assertEquals(best_individual_in_pbil.getGene(i), best_individual.getGene(i));
    }

  }

  /*
   * We create testing population consist of two binary individual and we tested
   * correctnes of getBestResult
   */
  public void testGetBestResult() {
    EvBinaryVectorPBILOperator pbil = new EvBinaryVectorPBILOperator(new EvOneMax(), 1, 0.34, 0.45,
        0.65, 3650, new EvBinaryVectorSpace(new EvOneMax(), 3650));

    EvPopulation<EvBinaryVectorIndividual> population_to_set = new EvPopulation<EvBinaryVectorIndividual>();

    EvBinaryVectorIndividual individual = new EvBinaryVectorIndividual(3);

    individual.setGene(0, 1);
    individual.setGene(1, 1);
    individual.setGene(2, 1);

    population_to_set.add(individual);

    individual = new EvBinaryVectorIndividual(3);

    individual.setGene(0, 0);
    individual.setGene(1, 1);
    individual.setGene(2, 1);

    population_to_set.add(individual);

    Class pbil_reflection = pbil.getClass();
    Field field_population = null;

    try {
      field_population = pbil_reflection.getDeclaredField("population");
      field_population.setAccessible(true);
      field_population.set(pbil, population_to_set);
    } catch (Exception ex) {
      System.out.println(ex);
      fail();
    }
    assertEquals(pbil.getBestResult(), population_to_set.get(0));

    pbil = new EvBinaryVectorPBILOperator(new EvOneMax(), 1, 0.34, 0.45, 0.65, 3650,
        new EvBinaryVectorSpace(new EvOneMax(), 3650));
    population_to_set = new EvPopulation<EvBinaryVectorIndividual>();

    individual = new EvBinaryVectorIndividual(3);
    individual.setGene(0, 0);
    individual.setGene(1, 0);
    individual.setGene(2, 0);
    population_to_set.add(individual);

    individual = new EvBinaryVectorIndividual(3);
    individual.setGene(0, 0);
    individual.setGene(1, 0);
    individual.setGene(2, 1);
    population_to_set.add(individual);

    try {
      field_population.set(pbil, population_to_set);
    } catch (Exception ex) {
      fail();
    }

    assertEquals(pbil.getBestResult(), population_to_set.get(1));
  }

}