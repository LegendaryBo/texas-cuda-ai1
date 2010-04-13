package pl.wroc.uni.ii.evolution.engine.operators.general.statistic;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.scatterplot.EvKNaryScatterPlotGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.scatterplot.EvScatterPlotStatistic;

/**
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvScatterPlotGathererTest  extends TestCase {

  /** in this test we create small population of knary individuals and
  * simply apply the tested operator to it.
  */
  public void testGatherer() {
    
    EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();
    
    EvPopulation<EvKnaryIndividual> population = new 
        EvPopulation<EvKnaryIndividual>();
    
    EvKnaryIndividual ind1 = new EvKnaryIndividual(new int[]{0, 1, 0, 1}, 2);
    EvKnaryIndividual ind2 = new EvKnaryIndividual(new int[]{1, 2, 1, 2}, 2);
    EvKnaryIndividual ind3 = new EvKnaryIndividual(new int[]{1, 1, 1, 1}, 2);
    EvKnaryIndividual ind4 = new EvKnaryIndividual(new int[]{2, 2, 2, 2}, 2);
    
    population.add(ind1);
    population.add(ind2);
    population.add(ind3);
    population.add(ind4);
    
    EvKNaryScatterPlotGatherer gatherer = new 
        EvKNaryScatterPlotGatherer(new int[] {0, 2}, storage);
    
    population = gatherer.apply(population);
    
    EvStatistic[] stats = (EvStatistic[]) storage.getStatistics();
    
    double[][] result = ((EvScatterPlotStatistic) stats[0]).getGenes();
    double[][] correct_answer = new double[4][2];
    
    correct_answer[0] = new double[]{0.0, 0.0};
    correct_answer[1] = new double[]{1.0, 1.0};
    correct_answer[2] = new double[]{1.0, 1.0};
    correct_answer[3] = new double[]{2.0, 2.0};
    
    for (int i = 0; i < correct_answer.length; i++) {
      for (int j = 0; j < correct_answer[i].length; j++) {
        assertEquals(correct_answer[i][j], result[i][j]); 
      }
    }
    
  }
  
}
