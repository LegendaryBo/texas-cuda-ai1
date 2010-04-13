package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.Date;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvHistoricalPricesFromYahoo;
import pl.wroc.uni.ii.evolution.solutionspaces.EvPortfolioSpace;

public class EvPortfolioSpaceTest extends TestCase {

  @SuppressWarnings("deprecation")
  public void testBelongsTo() {
    
    EvPortfolioSpace space = new EvPortfolioSpace(1000, new Date(2005, 1 , 1), new Date(2006, 1 , 1), 
        new String[] {"MSFT", "GOOG", "DELL", "CSCO"},
        new EvHistoricalPricesFromYahoo()); 

    EvNaturalNumberVectorIndividual ind = new EvNaturalNumberVectorIndividual(new int[] {0,0,0,0});
    assertTrue(space.belongsTo(ind));
    EvNaturalNumberVectorIndividual ind2 = new EvNaturalNumberVectorIndividual(new int[] {0,3,0,0});
 
    assertTrue(space.belongsTo(ind2));
    EvNaturalNumberVectorIndividual ind3 = new EvNaturalNumberVectorIndividual(new int[] {10, 1,0,0});
    assertTrue(space.belongsTo(ind3));
    
  }

  @SuppressWarnings("deprecation")
  public void testGenerateIndividual() {
    EvPortfolioSpace space = new EvPortfolioSpace(1000, new Date(2005, 1 , 1), new Date(2006, 1 , 1), 
        new String[] {"MSFT", "GOOG", "DELL", "CSCO"},
        new EvHistoricalPricesFromYahoo());

    
    for (int i = 1; i < 10; i++) {
      EvNaturalNumberVectorIndividual in = space.generateIndividual();
      
      assertTrue(in.toString(), space.belongsTo(in));
    }
    
  }

}
