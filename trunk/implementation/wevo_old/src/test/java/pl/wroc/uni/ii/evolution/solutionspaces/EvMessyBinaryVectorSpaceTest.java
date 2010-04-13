package pl.wroc.uni.ii.evolution.solutionspaces;

import junit.framework.TestCase;
import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvMessyBinaryVectorMaxSum;
import pl.wroc.uni.ii.evolution.solutionspaces.EvMessyBinaryVectorSpace;

/** 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvMessyBinaryVectorSpaceTest extends TestCase {
  
  public void testBelongsTo() {
    ArrayList<Integer> genes = new ArrayList<Integer>(2);
    ArrayList<Boolean> alleles = new ArrayList<Boolean>(2);
    genes.add(0);alleles.add(true);
    genes.add(3);alleles.add(false);
    
    EvMessyBinaryVectorIndividual individual1 =
      new EvMessyBinaryVectorIndividual(3, genes, alleles);    
    EvMessyBinaryVectorIndividual individual2 =
      new EvMessyBinaryVectorIndividual(4, genes, alleles);    
    EvMessyBinaryVectorIndividual individual3 =
      new EvMessyBinaryVectorIndividual(5, genes, alleles);    
      
    EvMessyBinaryVectorSpace messy_space =
      new EvMessyBinaryVectorSpace(new EvMessyBinaryVectorMaxSum(), 4);
    
    assertFalse(messy_space.belongsTo(individual1));
    assertTrue(messy_space.belongsTo(individual2));
    assertFalse(messy_space.belongsTo(individual3));
  }  

  
  public void testGenerateIndividual() {
    
    double[] probability_vector_true = {1.0, 0.0};
    double[] probability_vector_false = {0.0, 1.0};

    EvMessyBinaryVectorSpace messy_space =
      new EvMessyBinaryVectorSpace(new EvMessyBinaryVectorMaxSum(), 2);
    
    EvMessyBinaryVectorIndividual individual_true =
      messy_space.generateIndividual(probability_vector_true);
    EvMessyBinaryVectorIndividual individual_false =
      messy_space.generateIndividual(probability_vector_false);
    
    assertEquals(individual_true.getGenotypeLength(), 2);
    assertEquals(individual_true.getChromosomeLength(), 2);
    assertEquals(individual_false.getGenotypeLength(), 2);
    assertEquals(individual_false.getChromosomeLength(), 2);
    
    if (individual_true.getGene(0) == 0) {
      assertEquals(individual_true.getAllele(0).booleanValue(), true);
      assertEquals(individual_true.getAllele(1).booleanValue(), false);
    } else {
      assertEquals(individual_true.getAllele(0).booleanValue(), false);
      assertEquals(individual_true.getAllele(1).booleanValue(), true);
    }
    
    if (individual_false.getGene(0) != 0) {
      assertEquals(individual_false.getAllele(0).booleanValue(), true);
      assertEquals(individual_false.getAllele(1).booleanValue(), false);
    } else {
      assertEquals(individual_false.getAllele(0).booleanValue(), false);
      assertEquals(individual_false.getAllele(1).booleanValue(), true);
    }

    assertTrue(messy_space.belongsTo(individual_true));    
    assertTrue(messy_space.belongsTo(individual_false));    
  }
  
  
  public void testGenerateAllAlleleCombinationIndividual() {
    
    EvMessyBinaryVectorSpace messy_space =
        new EvMessyBinaryVectorSpace(new EvMessyBinaryVectorMaxSum(), 4);
    
    ArrayList<Integer> genes = new ArrayList<Integer>(4);
    genes.add(0);genes.add(1);genes.add(2);genes.add(3);
    
    List<EvMessyBinaryVectorIndividual> list =
        messy_space.generateAllAlleleCombinationIndividuals(genes, 1);
    
    // There should be 16 distinct individuals
    assertEquals(list.size(), 16);
    for (int i = 0; i < list.size() - 1; i++)
      for (int j = i + 1; j < list.size(); j++)
        assertFalse(list.get(i).equals(list.get(j)));
    
    // All individuals must belongs to the space
    for (int i = 0; i < list.size(); i++)
      assertTrue(messy_space.belongsTo(list.get(i)));    
  }
  
  
  public void testGenerateCoverIndividuals() {
    
    List<EvMessyBinaryVectorIndividual> list;
    EvMessyBinaryVectorSpace messy_space_7 =
        new EvMessyBinaryVectorSpace(new EvMessyBinaryVectorMaxSum(), 7);
    EvMessyBinaryVectorSpace messy_space_8 =
        new EvMessyBinaryVectorSpace(new EvMessyBinaryVectorMaxSum(), 8);
    boolean alleles_template[] = new boolean[8];
    for (int i = 0; i < 8; i++)
      alleles_template[i] = i % 2 != 0;
    
    /* No alleles template test */
    list = messy_space_7.generateCoverIndividuals(2, 1, 0);
    assertEquals(list.size(), 7 * 6 / 2 * (2 * 2));// 1 * (n 2) * 2^2
    // Check for distincts
    for (int i = 0; i < list.size() - 1; i++)
      for (int j = i + 1; j < list.size(); j++)
        assertFalse(list.get(i).equals(list.get(j)));
    // All individuals must belongs to the space
    for (int i = 0; i < list.size(); i++)
      assertTrue(messy_space_7.belongsTo(list.get(i)));    
    
    /* Alleles template test */
    list = messy_space_8.generateCoverIndividuals(3, 1, 0, alleles_template);
    assertEquals(list.size(), 8 * 7 * 6 / (3 * 2));// 1 * (n 3)
    // Check for distincts
    for (int i = 0; i < list.size() - 1; i++)
      for (int j = i + 1; j < list.size(); j++)
        assertFalse(list.get(i).equals(list.get(j)));
    // All individuals must belongs to the space
    for (int i = 0; i < list.size(); i++)
      assertTrue(messy_space_8.belongsTo(list.get(i)));
    
    /* No alleles template test with 2 copies */
    list = messy_space_8.generateCoverIndividuals(1, 2, 0);
    assertEquals(list.size(), 8 * 2 * 2);// 2 * (n 1) * 2^1 
    // All individuals must belongs to the space
    for (int i = 0; i < list.size(); i++)
      assertTrue(messy_space_8.belongsTo(list.get(i)));

    /* Maximum number of individuals test */
    list = messy_space_8.generateCoverIndividuals(2, 1, 30);
    assertEquals(list.size(), 30);// only 30 from 84
    // All individuals must belongs to the space
    for (int i = 0; i < list.size(); i++)
      assertTrue(messy_space_8.belongsTo(list.get(i)));
    // Check for distincts
    for (int i = 0; i < list.size() - 1; i++)
      for (int j = i + 1; j < list.size(); j++)
        assertFalse(list.get(i).equals(list.get(j)));
      
    /*
    // Efficienty test  
     EvMessyBinaryVectorSpace messy_space_50 =
      new EvMessyBinaryVectorSpace(new EvMessyBinaryVectorMaxSum(), 50);
    boolean alleles_template_50[] = new boolean[50];
    for (int i = 0; i < 50; i++)
      alleles_template_50[i] = i % 2 != 0;
    list = messy_space_50.generateCoverIndividuals(4, 1, 0, alleles_template_50);
    */
  }
  
}