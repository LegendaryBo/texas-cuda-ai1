package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import junit.framework.TestCase;
import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMessyPairsTournamentSelection;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvMessyBinaryVectorMaxSum;

/**
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvMessyPairsTournamentSelectionTest extends TestCase {

  //Checking thresholding and tie breaking (always stronger with weaker)
  public void testApply() {
    EvMessyPairsTournamentSelection<EvMessyBinaryVectorIndividual> selection =
        new EvMessyPairsTournamentSelection<EvMessyBinaryVectorIndividual>(
            2, true, true);
    EvObjectiveFunction<EvMessyBinaryVectorIndividual> objective_function =
        new EvMessyBinaryVectorMaxSum();
    
    EvPopulation<EvMessyBinaryVectorIndividual> population =
        new EvPopulation<EvMessyBinaryVectorIndividual>(4);
    ArrayList<Boolean> alleles2 = new ArrayList<Boolean>(2);
    ArrayList<Boolean> alleles3 = new ArrayList<Boolean>(3);
    alleles2.add(true);alleles2.add(true);
    alleles3.add(true);alleles3.add(true);alleles3.add(false);
    ArrayList<Integer> genes1 = new ArrayList<Integer>(2);
    ArrayList<Integer> genes2 = new ArrayList<Integer>(3);
    ArrayList<Integer> genes3 = new ArrayList<Integer>(2);
    ArrayList<Integer> genes4 = new ArrayList<Integer>(3);
    genes1.add(1);genes1.add(2);
    genes2.add(2);genes2.add(3);genes2.add(3);
    genes3.add(4);genes3.add(5);
    genes4.add(5);genes4.add(6);genes4.add(6);
    population.add(new EvMessyBinaryVectorIndividual(9, genes1, alleles2));
    population.add(new EvMessyBinaryVectorIndividual(9, genes2, alleles3));
    population.add(new EvMessyBinaryVectorIndividual(9, genes3, alleles2));
    population.add(new EvMessyBinaryVectorIndividual(9, genes4, alleles3));
    for(EvMessyBinaryVectorIndividual individual: population)
      individual.setObjectiveFunction(objective_function);
    
    for (int i = 0; i < 16; i++) {
      EvPopulation<EvMessyBinaryVectorIndividual> new_population =
          selection.apply(population);
      
      /* Thresholding for each pair of these indviduals is more than 0,
       * only 1, 2 and 3, 4 pair have 1 common expressed gene,
       * so they should be always fight together in thresholding tournament.
       * Also the 1st and 3rd individuals are shorter so they always win the
       * tournament due to tie breaking.
       * There must be 2 different individuals
       * with 2 chromosome length (1st and 3rd).
       */
      assertEquals(new_population.size(), 2);
      assertFalse(new_population.get(0).equals(new_population.get(1)));
      assertEquals(new_population.get(0).getChromosomeLength(), 2);
      assertEquals(new_population.get(1).getChromosomeLength(), 2);
    }
    
    //Checking thesholding above tie breaking (2 weaker, 2 stronger together)
    genes1.clear();genes1.add(1);genes1.add(2);
    genes2.clear();genes2.add(4);genes2.add(5);genes2.add(5);
    genes3.clear();genes3.add(2);genes3.add(3);
    genes4.clear();genes4.add(5);genes4.add(5);genes4.add(6);
    population.clear();
    population.add(new EvMessyBinaryVectorIndividual(9, genes1, alleles2));
    population.add(new EvMessyBinaryVectorIndividual(9, genes2, alleles3));
    population.add(new EvMessyBinaryVectorIndividual(9, genes3, alleles2));
    population.add(new EvMessyBinaryVectorIndividual(9, genes4, alleles3));
    for(EvMessyBinaryVectorIndividual individual: population)
      individual.setObjectiveFunction(objective_function);

    for (int i = 0; i < 16; i++) {
      EvPopulation<EvMessyBinaryVectorIndividual> new_population =
          selection.apply(population);
      
      /* Thresholding for each pair of these indviduals is more than 0,
       * only 1, 3 and 2, 4 pair have 1 common expressed gene,
       * so they should be always fight together in thresholding tournament.
       * There must be 2 different individuals,
       * one with 2 and one with 3 chromosome length.
       */
      assertEquals(new_population.size(), 2);
      assertFalse(new_population.get(0).equals(new_population.get(1)));
      assertTrue(
          (new_population.get(0).getChromosomeLength() == 2 &&
           new_population.get(1).getChromosomeLength() == 3) ||
          (new_population.get(0).getChromosomeLength() == 3 &&
           new_population.get(1).getChromosomeLength() == 2));
    }
    
  }
  
}