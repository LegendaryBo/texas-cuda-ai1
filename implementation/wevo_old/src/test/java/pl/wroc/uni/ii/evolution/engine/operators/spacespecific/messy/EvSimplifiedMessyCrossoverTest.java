package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import java.util.ArrayList;
import java.util.Arrays;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvSimplifiedMessyCrossover;

public class EvSimplifiedMessyCrossoverTest extends TestCase {

  public void testApply() {
    
    EvSimplifiedMessyCrossover operator = 
      new EvSimplifiedMessyCrossover(1.0);
    
    EvPopulation<EvSimplifiedMessyIndividual> population = new EvPopulation<EvSimplifiedMessyIndividual>();
    EvSimplifiedMessyIndividual father = new EvSimplifiedMessyIndividual(2);
    EvSimplifiedMessyIndividual mother = new EvSimplifiedMessyIndividual(2);
    father.addGeneValue(0, 1);
    father.addGeneValue(1, 7);
    father.addGeneValue(1, 8);

    mother.addGeneValue(0, 13);
    mother.addGeneValue(1, 20);
    father.addGeneValue(1, 23);
  
    
    population.add(father);
    population.add(mother);

    population = new EvPopulation<EvSimplifiedMessyIndividual>(operator.apply(population));
    
    System.out.println(population.get(0));
    System.out.println(population.get(1));
  
    
    ArrayList<Integer> tmp1 =  population.get(0).getGeneValues(1);
    ArrayList<Integer> tmp2 =  population.get(1).getGeneValues(1);
    
    ArrayList<Integer> sum = new ArrayList<Integer>();
    
    if (tmp1 != null) {
      sum.addAll(tmp1);
    }
    if (tmp2 != null) {
      sum.addAll(tmp2);
    }

    
    ArrayList<Integer> pattern = new ArrayList<Integer>();
    pattern.add(7);
    pattern.add(8);
    pattern.add(20);
    pattern.add(23);
    
    Integer[] t1 = convertObjectArray(sum.toArray());
    Integer[] t2 = convertObjectArray(pattern.toArray());
    
    Arrays.sort(t1);
    Arrays.sort(t2);
    
//    System.out.println(population.get(0));
//    System.out.println(population.get(1));
    
    for (int i=0; i < t1.length; i++) {
      //System.out.println(t1[i]);
      assertEquals(t1[i], t2[i]);
    }
    
    
    

  }
  private Integer[] convertObjectArray(Object[] array) {
    Integer[] tab = new Integer[array.length];
    
    for (int i = 0; i < array.length; i++) {
      tab[i] = (Integer) array[i];
    }
    
    
    return tab;
  }

}
