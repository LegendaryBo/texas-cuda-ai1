package pl.wroc.uni.ii.evolution.solutionspaces;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

public class EvBinaryStringsTest extends TestCase {

  private static final int tested_dimension = 10;

  public void testGenerateIndividual() {
    EvBinaryVectorSpace binary_strings = new EvBinaryVectorSpace(new EvOneMax(), tested_dimension);
    double[] probability_vector = new double[tested_dimension];

    /* Tests if can generate zeros. */
    for (int i = 0; i < tested_dimension; i++) {
      probability_vector[i] = 0.0;
    }
    EvBinaryVectorIndividual b = (EvBinaryVectorIndividual) binary_strings
        .generateIndividual(probability_vector);
    for (int i = 0; i < tested_dimension; i++) {
      assertEquals(b.getGene(i), 0);
    }

    /* Tests if can generate ones. */
    for (int i = 0; i < tested_dimension; i++) {
      probability_vector[i] = 1.0;
    }
    b = (EvBinaryVectorIndividual) binary_strings
        .generateIndividual(probability_vector);
    for (int i = 0; i < tested_dimension; i++) {
      assertEquals(b.getGene(i), 1);
    }
  }
}
