package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;
import junit.framework.TestCase;

/**
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvMessyGeneDeletionMutationTest extends TestCase {
  
  private final int chromosome_length = 50;
  private final int number_of_genes_to_delete = 10;

  /**
   * 
   * Generic class which is useful in tests
   *
   */
  private class MyTest <T extends EvMessyIndividual>{
    
    EvPopulation<T> tested_population;
    EvOperator<T> delete_genes_operator;
    
    public MyTest(EvPopulation<T> population) {
      tested_population = population;
      delete_genes_operator =
          new EvMessyGeneDeletionMutation<T>(
              number_of_genes_to_delete);
    }
    
    public void test() {
      EvPopulation<T> cut_population = 
          delete_genes_operator.apply(tested_population);
      T cut_individual = cut_population.get(0);
      int cut_individual_len = cut_individual.getChromosomeLength();
      assertEquals(chromosome_length, 
          cut_individual_len + number_of_genes_to_delete);
    }
  }
  
  /**
   * We are testing MessyBinaryVectorIndividual
   */
  public void testMessyBinaryVectorIndividual() {
    
    ArrayList<Integer> genes = new ArrayList<Integer>(chromosome_length);
    ArrayList<Boolean> alleles = new ArrayList<Boolean>(chromosome_length);
    
    for(int i = 0; i < chromosome_length; i++) {
      int pos = EvRandomizer.INSTANCE.nextInt(chromosome_length);
      boolean val = EvRandomizer.INSTANCE.nextBoolean();
      genes.add(pos);
      alleles.add(val);
    }
    
    // create some messy individuals
    EvMessyBinaryVectorIndividual individual = 
        new EvMessyBinaryVectorIndividual(
            chromosome_length, genes, alleles);

    EvPopulation<EvMessyBinaryVectorIndividual> population = 
        new EvPopulation<EvMessyBinaryVectorIndividual>();
    
    population.add(individual);
    MyTest<EvMessyBinaryVectorIndividual> my_test = 
        new MyTest<EvMessyBinaryVectorIndividual>(population);
    my_test.test();
  }
  
  /**
   * We are testing OmegaIndividual
   */
  public void testOmegaIndividual() {
    ArrayList<Integer> genes = new ArrayList<Integer>(chromosome_length);
    ArrayList<Double> alleles = new ArrayList<Double>(chromosome_length);
    
    for(int i = 0; i < chromosome_length; i++) {
      int pos = EvRandomizer.INSTANCE.nextInt(chromosome_length);
      double val = EvRandomizer.INSTANCE.nextDouble();
      genes.add(pos);
      alleles.add(val);
    }
    // create some messy individuals
    EvOmegaIndividual individual = 
        new EvOmegaIndividual(
            chromosome_length, genes, alleles, null);
    
    EvPopulation<EvOmegaIndividual> population = 
        new EvPopulation<EvOmegaIndividual>();
    
    population.add(individual);

    MyTest<EvOmegaIndividual> my_test = 
        new MyTest<EvOmegaIndividual>(population);
    
    my_test.test();
  }
  
}
